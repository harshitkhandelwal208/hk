// Reads the fixture written by tests/bindings/c_abi.c and exercises the writer.
// Run: HK_FIXTURE=<fixture.hk> LD_LIBRARY_PATH=<dir with libhk> dotnet run --project bindings/csharp/tests
using System;
using System.IO;
using Hk;

static class Program
{
    static int failures;

    static void Check(bool ok, string what)
    {
        if (!ok)
        {
            failures++;
            Console.Error.WriteLine("FAIL " + what);
        }
    }

    static int Main()
    {
        string? fixture = Environment.GetEnvironmentVariable("HK_FIXTURE");
        if (fixture == null)
        {
            Console.WriteLine("HK_FIXTURE not set, skipping");
            return 0;
        }

        using (var m = HkModel.Open(fixture))
        {
            Check(m.TensorCount == 2, "tensor count");
            var t = m.GetTensor(0);
            Check(t.Name == "w.f32" && t.StorageType == StorageType.F32, "tensor 0 identity");
            Check(t.Ndim == 2 && t.Shape[0] == 2 && t.Shape[1] == 3 && t.ElementCount == 6, "shape");
            var f = t.Dequantize();
            Check(f[0] == 1f && f[5] == 6f, "f32 values");
            var q = m.GetTensor(1);
            Check(q.StorageType == StorageType.Q8_0, "q8 type");
            Check(Math.Abs(q.Dequantize(false)[0] + 4f) < 0.02f, "q8 values");

            Check(m.GetMetadataString("general.name") == "fixture", "string metadata");
            Check(m.GetMetadataInt("answer") == 42, "int metadata");
            Check(m.GetMetadataFloat("pi") == 3.5, "float metadata");
            Check(m.GetMetadataBool("flag") == true, "bool metadata");
            Check(m.GetMetadataInt("missing") == null, "missing metadata");
            Check(m.FileAlignment == 4096 && m.IsUniversalPageAligned && !m.IsSharded, "header flags");

            Check(m.AppendixCount == 1, "appendix count");
            var e = m.GetAppendixEntry(0);
            Check(e.Name == "gen1" && e.Target == "w.f32" && e.Generation == 1, "appendix names");
            Check(System.Text.Encoding.UTF8.GetString(e.Data) == "adapter-bytes", "appendix payload");
        }

        string dir = Path.Combine(Path.GetTempPath(), "hk-cs-" + Environment.ProcessId);
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, "w.hk");
        using (var w = new HkWriter(128))
        {
            w.AddMetadataString("k", "v");
            var data = new byte[8];
            BitConverter.GetBytes(1.5f).CopyTo(data, 0);
            BitConverter.GetBytes(2.5f).CopyTo(data, 4);
            w.AddTensor("t", StorageType.F32, TileLayout.RowMajor, SparsityType.None, new ulong[] { 2 }, data, 0f);
            w.WriteToFile(path);
        }
        Check(HkModel.PatchMetadataInPlace(path, "k", "changed"), "patch");
        using (var m = HkModel.Open(path))
        {
            Check(m.GetMetadataString("k") == "changed", "patched value");
            var v = m.GetTensor(0).Dequantize(false);
            Check(v[0] == 1.5f && v[1] == 2.5f, "written values");
        }
        Directory.Delete(dir, true);

        Check(HkModel.DetectHardware().OptimalPageAlignment >= 128, "hardware");

        Console.WriteLine(failures == 0 ? "csharp ok" : failures + " failures");
        return failures == 0 ? 0 : 1;
    }
}
