//! Reads the fixture written by tests/bindings/c_abi.c. Set HK_FIXTURE to its path and HK_LIB_DIR
//! to the directory with libhk. Without HK_FIXTURE the tests that need it are skipped.

use hknt::*;

fn fixture() -> Option<String> {
    std::env::var("HK_FIXTURE").ok()
}

#[test]
fn reads_fixture() {
    let Some(path) = fixture() else { return };
    let m = HkModel::open(&path).unwrap();
    assert_eq!(m.tensor_count(), 2);
    let t = m.get_tensor(0).unwrap();
    assert_eq!(t.name(), "w.f32");
    assert_eq!(t.storage_type(), StorageType::F32);
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.as_raw_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let q = m.get_tensor(1).unwrap();
    assert_eq!(q.storage_type(), StorageType::Q8_0);
    let d = q.dequantize(false).unwrap();
    assert_eq!(d.len(), 64);
    assert!((d[0] + 4.0).abs() < 0.02);

    assert_eq!(m.get_metadata_string("general.name").as_deref(), Some("fixture"));
    assert_eq!(m.get_metadata_int("answer"), Some(42));
    assert_eq!(m.get_metadata_float("pi"), Some(3.5));
    assert_eq!(m.get_metadata_bool("flag"), Some(true));
    assert_eq!(m.get_metadata_int("missing"), None);
    assert_eq!(m.file_alignment(), 4096);
    assert!(m.is_universal_page_aligned());
    assert!(!m.is_sharded());

    assert_eq!(m.appendix_count(), 1);
    let e = m.get_appendix_entry(0).unwrap();
    assert_eq!(e.name, "gen1");
    assert_eq!(e.target, "w.f32");
    assert_eq!(e.generation, 1);
    assert_eq!(e.data, b"adapter-bytes");
}

#[test]
fn writer_round_trip_and_rollback() {
    let dir = std::env::temp_dir().join(format!("hk-rust-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("w.hk");
    let path = path.to_str().unwrap();
    {
        let mut w = HkWriter::new(128).unwrap();
        w.add_metadata_string("k", "v").unwrap();
        w.add_metadata_bool("b", true).unwrap();
        let data: Vec<u8> = [1.5f32, 2.5, 3.5, 4.5].iter().flat_map(|f| f.to_le_bytes()).collect();
        w.add_tensor("t", StorageType::F32, TileLayout::RowMajor, SparsityType::None, &[2, 2], &data, 0.0).unwrap();
        w.write_to_file(path).unwrap();
    }
    let m = HkModel::open(path).unwrap();
    assert_eq!(m.get_metadata_string("k").as_deref(), Some("v"));
    assert_eq!(m.get_metadata_bool("b"), Some(true));
    assert_eq!(m.get_tensor(0).unwrap().dequantize(false).unwrap(), vec![1.5, 2.5, 3.5, 4.5]);
    drop(m);

    HkModel::patch_metadata_in_place(path, "k", "changed").unwrap();
    let m = HkModel::open(path).unwrap();
    assert_eq!(m.get_metadata_string("k").as_deref(), Some("changed"));
    assert_eq!(m.get_tensor(0).unwrap().dequantize(false).unwrap()[0], 1.5);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn math_helpers() {
    assert_eq!(dot_f32(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    let mut y = [0.0f32; 2];
    gemv_f32(&[1.0, 0.0, 0.0, 1.0], &[3.0, 4.0], None, &mut y, 2, 2);
    assert_eq!(y, [3.0, 4.0]);
    assert!(detect_hardware().optimal_page_alignment >= 128);
}
