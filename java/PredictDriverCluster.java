import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.net.URI;

public class PredictDriverCluster {

    public static void main(String[] args) throws Exception {

        // --- Kiểm tra tham số đầu vào ---
        if (args.length != 2) {
            System.err.println("Usage: PredictDriverCluster <input_list.txt (HDFS)> <output_dir (HDFS)>");
            System.exit(1);
        }

        // --- Cấu hình Hadoop Job ---
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Few-Shot Batch Prediction Cluster");

        // --- Đặt class chính ---
        job.setJarByClass(PredictDriverCluster.class);

        // --- Cấu hình Mapper & Reducer ---
        job.setMapperClass(PredictBatchMapperCluster.class);
        job.setReducerClass(PredictReducerCSV.class);

        // --- Kiểu dữ liệu đầu ra ---
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        // --- Định dạng input/output ---
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        // --- Đường dẫn input/output ---
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        // --- Distributed Cache ---
        // Python script
        job.addCacheFile(new URI("/user/binh/garbage/predict_batch_threaded_local.py#predict_batch_threaded_local1.py"));
        // Checkpoint model
        job.addCacheFile(new URI("/user/binh/garbage/stage3_model_best-epoch_10_best.pth#checkpoint.pth"));
        // Folder chứa class zip (nếu file zip này được giải nén trong mapper)
        job.addCacheArchive(new URI("/user/binh/garbage/class_folder.zip#class_folder"));

        // --- Submit job ---
        boolean success = job.waitForCompletion(true);
        System.exit(success ? 0 : 1);
    }
}
