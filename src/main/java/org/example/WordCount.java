package org.example;

import java.io.IOException;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import java.util.HashMap;
import java.util.Map;

public class WordCount {

    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {

                String cleanedWord = itr.nextToken().replaceAll("[^a-zA-Z]", "").toLowerCase();

                if (!cleanedWord.isEmpty()) {
                    word.set(cleanedWord);
                    context.write(word, one);
                }
            }
        }
    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

        private TreeMap<String, Integer> sortedWords = new TreeMap<>();
        private List<String> mostFrequentWords = new ArrayList<>();
        private int maxFrequency = Integer.MIN_VALUE;
        private List<String> leastFrequentWords = new ArrayList<>();
        private Map<Integer, List<String>> frequencyToWordsMap = new HashMap<>();
        private int minFrequency = Integer.MAX_VALUE;

        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }

            sortedWords.put(key.toString(), sum);

            // Update least frequent word(s) and reset the list
            if (sum < minFrequency) {
                leastFrequentWords.clear();
                leastFrequentWords.add(key.toString());
                minFrequency = sum;
            } else if (sum > maxFrequency) {
                mostFrequentWords.clear();
                mostFrequentWords.add(key.toString());
                maxFrequency = sum;
            }
            else if (sum == minFrequency) {
                leastFrequentWords.add(key.toString());
            }

            // Update the map of frequency to words
            frequencyToWordsMap.computeIfAbsent(sum, k -> new ArrayList<>()).add(key.toString());
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            // Write the sorted words alphabetically
            for (Map.Entry<String, Integer> entry : sortedWords.entrySet()) {
                context.write(new Text(entry.getKey()), new IntWritable(entry.getValue()));
            }

            // Write the least frequent word(s)
            context.write(new Text("Word with smallest frequency: " + leastFrequentWords), new IntWritable(minFrequency));
            context.write(new Text("Word with largest frequency: " + mostFrequentWords), new IntWritable(maxFrequency));


            // Write the words with the same frequency
            for (Map.Entry<Integer, List<String>> entry : frequencyToWordsMap.entrySet()) {
                int frequency = entry.getKey();
                List<String> words = entry.getValue();
                context.write(new Text("Words with frequency " + frequency + ": " + words), new IntWritable(frequency));
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "word count");

        Path outputDir = new Path(args[1]);
        FileSystem fs = FileSystem.get(conf);
        if (fs.exists(outputDir)) {
            fs.delete(outputDir, true);
        }

        job.setJarByClass(WordCount.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, outputDir);
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
