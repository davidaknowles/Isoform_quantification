//! radpos: dump the alignments of a single-cell RAD file whose records carry
//! the alignment-level tags `compressed_ori_refid`, `as`, `start`, `end` and
//! `tlen` (libradicl's ScLongRead record; written by the patched salmon
//! `alevin --rad`). Works on uncollated (`map.rad`) and collated
//! (`map.collated.rad`) files.
//!
//! Output directory, all little-endian flat arrays (numpy `fromfile`/`memmap`):
//!   read_bc.u64     cell barcode, 2-bit packed (A=0 C=1 G=2 T=3, first base highest)
//!   read_umi.u64    UMI, 2-bit packed
//!   read_naln.u32   number of alignments kept for the read
//!   aln_ref.u32     reference (target) index into refs.txt
//!   aln_fw.u8       1 if the alignment is forward
//!   aln_score.i32   alignment score
//!   aln_start.u32   leftmost transcript position of the read
//!   aln_end.u32     start + read length, clipped to the transcript length
//!   aln_tlen.u32    transcript length
//!   refs.txt        reference names
//!   meta.json       barcode and UMI lengths, record and alignment counts
//!
//! Usage: radpos <in.rad> <out_dir> [--fw-only]
//!   --fw-only keeps forward alignments only and drops reads left without any.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Cursor, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use libradicl::codec::{decompress_payload, ChunkCodec, CHUNK_CODEC_TAG};
use libradicl::header::RadPrelude;
use libradicl::rad_types::TagValue;
use libradicl::record::{MappedRecord, ScLongReadRecord, ScLongReadRecordContext};

struct Columns {
    bc: BufWriter<File>,
    umi: BufWriter<File>,
    naln: BufWriter<File>,
    rf: BufWriter<File>,
    fw: BufWriter<File>,
    score: BufWriter<File>,
    start: BufWriter<File>,
    end: BufWriter<File>,
    tlen: BufWriter<File>,
    n_reads: u64,
    n_alns: u64,
}

fn open(dir: &Path, name: &str) -> Result<BufWriter<File>> {
    let p = dir.join(name);
    Ok(BufWriter::with_capacity(1 << 20, File::create(&p).with_context(|| format!("creating {:?}", p))?))
}

impl Columns {
    fn new(dir: &Path) -> Result<Self> {
        Ok(Self {
            bc: open(dir, "read_bc.u64")?,
            umi: open(dir, "read_umi.u64")?,
            naln: open(dir, "read_naln.u32")?,
            rf: open(dir, "aln_ref.u32")?,
            fw: open(dir, "aln_fw.u8")?,
            score: open(dir, "aln_score.i32")?,
            start: open(dir, "aln_start.u32")?,
            end: open(dir, "aln_end.u32")?,
            tlen: open(dir, "aln_tlen.u32")?,
            n_reads: 0,
            n_alns: 0,
        })
    }

    fn push(&mut self, rec: &ScLongReadRecord, fw_only: bool) -> Result<()> {
        let keep: Vec<usize> = (0..rec.refs.len()).filter(|&i| !fw_only || rec.dirs[i]).collect();
        if keep.is_empty() {
            return Ok(());
        }
        self.bc.write_all(&rec.bc.to_le_bytes())?;
        self.umi.write_all(&rec.umi.to_le_bytes())?;
        self.naln.write_all(&(keep.len() as u32).to_le_bytes())?;
        for &i in &keep {
            self.rf.write_all(&rec.refs[i].to_le_bytes())?;
            self.fw.write_all(&[rec.dirs[i] as u8])?;
            self.score.write_all(&rec.as_scores[i].to_le_bytes())?;
            self.start.write_all(&rec.starts[i].to_le_bytes())?;
            self.end.write_all(&rec.ends[i].to_le_bytes())?;
            self.tlen.write_all(&rec.tlens[i].to_le_bytes())?;
        }
        self.n_reads += 1;
        self.n_alns += keep.len() as u64;
        Ok(())
    }

    fn finish(mut self) -> Result<(u64, u64)> {
        for w in [&mut self.bc, &mut self.umi, &mut self.naln, &mut self.rf, &mut self.fw,
                  &mut self.score, &mut self.start, &mut self.end, &mut self.tlen] {
            w.flush()?;
        }
        Ok((self.n_reads, self.n_alns))
    }
}

fn tag_u16(v: Option<&TagValue>, name: &str) -> Result<u64> {
    match v {
        Some(TagValue::U8(x)) => Ok(*x as u64),
        Some(TagValue::U16(x)) => Ok(*x as u64),
        Some(TagValue::U32(x)) => Ok(*x as u64),
        _ => bail!("file tag '{name}' missing or not an unsigned integer"),
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: radpos <in.rad> <out_dir> [--fw-only]");
    }
    let input = PathBuf::from(&args[1]);
    let out = PathBuf::from(&args[2]);
    let fw_only = args.iter().skip(3).any(|a| a == "--fw-only");
    fs::create_dir_all(&out)?;

    let mut br = BufReader::with_capacity(1 << 22, File::open(&input).with_context(|| format!("opening {:?}", input))?);
    let prelude = RadPrelude::from_bytes(&mut br)?;
    let aln_tags: Vec<String> = prelude.aln_tags.tags.iter().map(|t| t.name.clone()).collect();
    for want in ["compressed_ori_refid", "as", "start", "end", "tlen"] {
        if !aln_tags.iter().any(|t| t == want) {
            bail!("alignment tag '{want}' not in RAD file (tags: {:?}); not a position-carrying RAD", aln_tags);
        }
    }
    let file_tags = prelude.file_tags.parse_tags_from_bytes(&mut br)?;
    let cblen = tag_u16(file_tags.get("cblen"), "cblen")?;
    let ulen = tag_u16(file_tags.get("ulen"), "ulen")?;
    let codec = match file_tags.get(CHUNK_CODEC_TAG) {
        None => ChunkCodec::None,
        Some(TagValue::U8(v)) => ChunkCodec::from_u8(*v)?,
        Some(_) => bail!("'{CHUNK_CODEC_TAG}' file tag must be a U8"),
    };
    let ctx = prelude.get_record_context::<ScLongReadRecordContext>()?;

    let mut refs = BufWriter::new(File::create(out.join("refs.txt"))?);
    for r in &prelude.hdr.ref_names {
        writeln!(refs, "{r}")?;
    }
    refs.flush()?;

    let mut cols = Columns::new(&out)?;
    let mut n_chunks: u64 = 0;
    let mut n_records: u64 = 0;
    let mut hdr = [0u8; 8];
    loop {
        if br.fill_buf()?.is_empty() {
            break;
        }
        br.read_exact(&mut hdr)?;
        let nbytes = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        let nrec = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
        if codec == ChunkCodec::None {
            for _ in 0..nrec {
                let rec = ScLongReadRecord::from_bytes_with_context(&mut br, &ctx);
                cols.push(&rec, fw_only)?;
            }
        } else {
            let mut payload = vec![0u8; nbytes as usize - 8];
            br.read_exact(&mut payload)?;
            let raw = decompress_payload(codec, &payload)?;
            let mut cur = Cursor::new(raw.as_slice());
            for _ in 0..nrec {
                let rec = ScLongReadRecord::from_bytes_with_context(&mut cur, &ctx);
                cols.push(&rec, fw_only)?;
            }
        }
        n_chunks += 1;
        n_records += nrec as u64;
    }
    let (n_reads, n_alns) = cols.finish()?;
    let mut meta = File::create(out.join("meta.json"))?;
    write!(
        meta,
        "{{\"source\": {:?}, \"cblen\": {cblen}, \"ulen\": {ulen}, \"n_chunks\": {n_chunks}, \
         \"n_records\": {n_records}, \"n_reads\": {n_reads}, \"n_alns\": {n_alns}, \"fw_only\": {fw_only}, \
         \"n_refs\": {}}}\n",
        input.to_string_lossy(),
        prelude.hdr.ref_names.len()
    )?;
    eprintln!("radpos: {n_chunks} chunks, {n_records} records, kept {n_reads} reads with {n_alns} alignments");
    Ok(())
}
