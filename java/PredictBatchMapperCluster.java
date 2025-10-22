import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.*;
import java.util.regex.Pattern;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;

public class PredictBatchMapperCluster extends Mapper<LongWritable, Text, Text, Text> {

    private static final String PYTHON_SCRIPT_LOCAL = "predict_batch_threaded_local1.py";
    private static final String CKPT_LOCAL = "checkpoint.pth";
    private static final String CLASS_FOLDER_LOCAL = "class_folder";

    // Số ảnh xử lý trong mỗi batch
    private static final int BATCH_SIZE = 32;

    private java.nio.file.Path localTmpDir;
    private List<String> imageBatch = new ArrayList<>();
    private static final Pattern SAFE_FILENAME = Pattern.compile("[^A-Za-z0-9._-]");

    @Override
    protected void setup(Context context) throws IOException {
        localTmpDir = Files.createTempDirectory("predict_mapper_");

        File py = new File(PYTHON_SCRIPT_LOCAL);
        File ckpt = new File(CKPT_LOCAL);
        File cls = new File(CLASS_FOLDER_LOCAL);

        System.err.println(">>> Mapper setup at: " + new File(".").getAbsolutePath());
        if (!py.exists() || !ckpt.exists() || !cls.exists()) {
            System.err.println("⚠️ Missing cached files!");
            if (!py.exists()) System.err.println(" - Missing " + PYTHON_SCRIPT_LOCAL);
            if (!ckpt.exists()) System.err.println(" - Missing " + CKPT_LOCAL);
            if (!cls.exists()) System.err.println(" - Missing " + CLASS_FOLDER_LOCAL);
        } else {
            System.err.println("✅ All cached files found.");
        }
    }

    @Override
    protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {

        String hdfsImagePath = value.toString().trim();
        if (hdfsImagePath.isEmpty()) return;

        Configuration conf = context.getConfiguration();
        FileSystem fs = FileSystem.get(conf);
        Path src = new Path(hdfsImagePath);

        // Đặt tên file an toàn
        String safeName = SAFE_FILENAME.matcher(src.getName()).replaceAll("_");
        java.nio.file.Path localDest = localTmpDir.resolve(safeName);

        // Copy ảnh về local
        fs.copyToLocalFile(false, src, new Path(localDest.toString()), true);
        imageBatch.add(localDest.toString());

        // Khi đủ batch thì chạy predict
        if (imageBatch.size() >= BATCH_SIZE) {
            runBatchPrediction(context);
            imageBatch.clear();
        }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        if (!imageBatch.isEmpty()) {
            runBatchPrediction(context);
            imageBatch.clear();
        }

        // Xóa thư mục tạm
        try {
            Files.walk(localTmpDir)
                    .sorted(Comparator.reverseOrder())
                    .map(java.nio.file.Path::toFile)
                    .forEach(File::delete);
        } catch (Exception ignored) {}
    }

    private void runBatchPrediction(Context context)
            throws IOException, InterruptedException {

        if (imageBatch.isEmpty()) return;

        // Ghi danh sách ảnh batch
        java.nio.file.Path listFile = localTmpDir.resolve("image_list.txt");
        Files.write(listFile, imageBatch, StandardCharsets.UTF_8);

        System.err.println("🚀 Running prediction for batch of size: " + imageBatch.size());

        ProcessBuilder pb = new ProcessBuilder(
                "python", PYTHON_SCRIPT_LOCAL,
                listFile.toString(),
                CKPT_LOCAL,
                CLASS_FOLDER_LOCAL,
                "vitb32_openclip_laion400m",
                "cpu"
        );
        pb.redirectErrorStream(false); // giữ stderr riêng

        Process process = pb.start();

        // Đọc stdout của Python → chỉ lấy dòng image_path,class,prob
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;
                String[] parts = line.split(",");
                if (parts.length == 3) {
                    context.write(new Text(parts[0]), new Text(parts[1] + "," + parts[2]));
                }
            }
        }

        // Đọc stderr nếu muốn log ra console
        try (BufferedReader errReader = new BufferedReader(new InputStreamReader(process.getErrorStream()))) {
            String errLine;
            while ((errLine = errReader.readLine()) != null) {
                System.err.println(errLine); // chỉ in ra console, không ảnh hưởng Hadoop output
            }
        }

        int exitCode = process.waitFor();
        System.err.println("✅ Batch prediction exited with code: " + exitCode);

        // Cleanup các file tạm
        Files.deleteIfExists(listFile);
        for (String path : imageBatch) {
            Files.deleteIfExists(java.nio.file.Paths.get(path));
        }
    }
}
