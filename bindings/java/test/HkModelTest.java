import com.hk.HkModel;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;

/** Reads the fixture written by tests/bindings/c_abi.c and exercises the writer. */
public final class HkModelTest {
    private static int failures = 0;

    private static void check(boolean ok, String what) {
        if (!ok) {
            failures++;
            System.err.println("FAIL " + what);
        }
    }

    public static void main(String[] args) throws Exception {
        String fixture = System.getenv("HK_FIXTURE");
        if (fixture == null) {
            System.out.println("HK_FIXTURE not set, skipping");
            return;
        }
        try (HkModel m = HkModel.open(fixture)) {
            check(m.getTensorCount() == 2, "tensor count");
            HkModel.HkTensor t = m.getTensor(0);
            check(t.getName().equals("w.f32") && t.getStorageType() == HkModel.StorageType.F32, "tensor 0");
            long[] shape = t.getShape();
            check(shape.length == 2 && shape[0] == 2 && shape[1] == 3 && t.getElementCount() == 6, "shape");
            FloatBuffer f = t.dequantize(true);
            check(f.get(0) == 1f && f.get(5) == 6f, "f32 values");
            HkModel.HkTensor q = m.getTensor(1);
            check(q.getStorageType() == HkModel.StorageType.Q8_0, "q8 type");
            check(Math.abs(q.dequantize(false).get(0) + 4f) < 0.02f, "q8 values");
            check(t.getRawData().remaining() == 24, "raw data");

            check("fixture".equals(m.getMetadataString("general.name")), "string metadata");
            check(m.getMetadataInt("answer") == 42, "int metadata");
            check(m.getMetadataFloat("pi") == 3.5, "float metadata");
            check(m.getMetadataBool("flag"), "bool metadata");
            check(m.getFileAlignment() == 4096 && m.isUniversalPageAligned() && !m.isSharded(), "header flags");

            check(m.getAppendixCount() == 1, "appendix count");
            HkModel.HkAppendixEntry e = m.getAppendixEntry(0);
            check("gen1".equals(e.name) && "w.f32".equals(e.target) && e.generation == 1 && e.dataSize == 13, "appendix entry");
        }

        Path dir = Files.createTempDirectory("hk-java");
        String path = new File(dir.toFile(), "w.hk").getPath();
        try (HkModel.HkWriter w = new HkModel.HkWriter(128)) {
            w.addMetadataString("k", "v");
            ByteBuffer data = ByteBuffer.allocateDirect(8).order(ByteOrder.LITTLE_ENDIAN);
            data.putFloat(1.5f).putFloat(2.5f);
            data.flip();
            w.addTensor("t", HkModel.StorageType.F32, HkModel.TileLayout.ROW_MAJOR, HkModel.SparsityType.NONE, new long[] {2}, data, 0f);
            w.writeToFile(path);
        }
        check(HkModel.patchMetadataInPlace(path, "k", "changed"), "patch");
        try (HkModel m = HkModel.open(path)) {
            check("changed".equals(m.getMetadataString("k")), "patched value");
            FloatBuffer v = m.getTensor(0).dequantize(false);
            check(v.get(0) == 1.5f && v.get(1) == 2.5f, "written values");
        }
        new File(path).delete();
        dir.toFile().delete();

        System.out.println(failures == 0 ? "java ok" : failures + " failures");
        System.exit(failures == 0 ? 0 : 1);
    }
}
