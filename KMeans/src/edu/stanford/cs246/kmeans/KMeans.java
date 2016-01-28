package edu.stanford.cs246.kmeans;

import java.io.IOException;
import java.util.*;
import java.util.Map.Entry;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;

import static java.lang.Math.sqrt;
import static java.lang.Math.pow;
import static java.lang.Math.abs;

import org.apache.commons.lang.*;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import edu.stanford.cs246.kmeans.KMeans;

import java.util.*;

public class KMeans {
	
	public static float DistE(float[] a, float[] b) {
		float dist = 0;
		for (int i = 0; i < a.length; i++) {
			dist += pow((double)(a[i] - b[i]), 2);
		}
//		return (float)sqrt(dist);
		return dist;
	}
	public static float DistM(float[] a, float[] b) {
		float dist = 0;
		for (int i = 0; i < a.length; i++) {
			dist += abs((double)(a[i] - b[i]));
		}
		return dist;
	}
	
	public class FloatArrayWritable extends ArrayWritable {
		public FloatArrayWritable() {
			super(FloatWritable.class);
		}
	}
	static float loss_sum = 0;
	public static class Map extends
			Mapper<LongWritable, Text, Text, Text> {
		
		private Text cluster = new Text();
		private Text point_text = new Text();
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
//			ArrayWritable point_wr = new ArrayWritable(FloatWritable.class);
			Configuration conf = context.getConfiguration();
			String cent_loc = conf.get("cent_loc");
//			System.out.println(cent_loc);
			float[][] center = ReadCenter(cent_loc);
			//System.out.println(value);
			String[] parts = value.toString().split(" ");
//			FloatWritable[] point_array = new FloatWritable[parts.length];
			float[] point = new float[parts.length];
			String point_str = "";
            for (int j = 0; j < parts.length; j++) {
            	point[j] = Float.parseFloat(parts[j]);            	
            	point_str = point_str + Float.toString(point[j]) + " ";
            }
            // compute distance with each center and assign cluster
            float dist;
            float dist_min = 0;
            int loc = 0;
            for (int i = 0; i < 10; i++) {
            	dist = DistM(point, center[i]);
            	if (i == 0) {
            		dist_min = dist;
            	}
            	if (dist < dist_min) {
            		loc = i;
            		dist_min = dist;
            	}
            }
            cluster.set(""+loc);
            point_str = point_str + dist_min;
//            System.out.println(point_str);
            point_text.set(point_str);
//            point_wr.set(point_array);
//            System.out.println(loc);
            context.write(cluster, point_text);  
            
		}
	}
	
 
    public static class Reduce extends Reducer<Text, Text, Text, Text> {
    	private Text center_text = new Text();
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        	float[] center = new float[58];
        	float loss = 0;
        	for ( int i = 0; i < 58; i++) {center[i] = 0;}
        	int count = 0;
			for (Text val : values) {
//                FloatWritable[] point_array = (FloatWritable[]) val.get();
				String[] parts = val.toString().split(" ");
				for (int i = 0; i < 58; i++) {
					center[i] += Float.parseFloat(parts[i]);
				}
				count += 1;
				loss = loss + Float.parseFloat(parts[58]);
			}
			loss_sum = loss_sum + loss;
//			System.out.println(loss_sum);
            for (int i = 0; i < 58; i++ ) {center[i] = center[i] / count;}
            String text = "";
            for (int i = 0; i <58; i++) {
            	text = text + Float.toString(center[i]) + " ";
            }
            center_text.set(text);
            context.write(key, center_text);
            
        }
    }
    

	
	public static float[][] ReadCenter(String path) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(path));        
        float[][] c1 = new float[10][58];
        for (int i = 0; i < 10; i++) {
        	String line = br.readLine();
        	String[] parts = line.split(" ");
/*        	if (parts[0].split("\t").length > 1) {
        		parts[0] = parts[0].split("\t")[1];
        		System.out.println(parts.length);
        	}*/
            for (int j = 0; j < 58; j++) {
 //           	System.out.println(parts[j]);
            	c1[i][j] = Float.parseFloat(parts[j].replaceAll("[0-9]\t", ""));
            }
        }
        br.close();
        return c1;
	}
	public static void WriteCenter(float[][] center, String path) throws IOException {
		BufferedWriter bw = new BufferedWriter(new FileWriter(path));        
        for (int i = 0; i < center.length; i++) {
            for (int j = 0; j < center[0].length; j++) {
            	bw.write(String.valueOf(center[i][j]) + " ");
            }
            bw.write("\n");
        }
        bw.close();
	}
 
	class Center{ float[] center; }
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();

        BufferedWriter bw = new BufferedWriter(new FileWriter("loss_c1_M.txt")); 
		for (int kk = 0; kk < 20; kk++) {
			loss_sum = 0;
			int kk_1 = kk - 1;
			if (kk == 0) {
				conf.set("cent_loc", "c1.txt");
			}
			
			else {conf.set("cent_loc", "output_c1_M_"+kk_1+"/part-r-00000");}
			Job job = new Job(conf, "job");
			job.setJarByClass(KMeans.class);
			job.setOutputKeyClass(Text.class);
			job.setOutputValueClass(Text.class);

			job.setMapperClass(Map.class);
			job.setReducerClass(Reduce.class);

			job.setInputFormatClass(TextInputFormat.class);
			job.setOutputFormatClass(TextOutputFormat.class);
			job.setMapOutputValueClass(Text.class);
			
			
			FileInputFormat.addInputPath(job, new Path("data.txt"));
			
			
			FileOutputFormat.setOutputPath(job, new Path("output_c1_M_"+kk));

			job.waitForCompletion(true);
			bw.write(loss_sum + "\n");
		}
		bw.close();

		
		
		BufferedWriter bw2 = new BufferedWriter(new FileWriter("loss_c2_M.txt")); 
		for (int kk = 0; kk < 20; kk++) {
			loss_sum = 0;
			int kk_1 = kk - 1;
			if (kk == 0) {
				conf.set("cent_loc", "c2.txt");
			}
			
			else {conf.set("cent_loc", "output_c2_M_"+kk_1+"/part-r-00000");}
			Job job = new Job(conf, "job");
			job.setJarByClass(KMeans.class);
			job.setOutputKeyClass(Text.class);
			job.setOutputValueClass(Text.class);

			job.setMapperClass(Map.class);
			job.setReducerClass(Reduce.class);

			job.setInputFormatClass(TextInputFormat.class);
			job.setOutputFormatClass(TextOutputFormat.class);
			job.setMapOutputValueClass(Text.class);
			
			
			FileInputFormat.addInputPath(job, new Path("data.txt"));
			
			
			FileOutputFormat.setOutputPath(job, new Path("output_c2_M_"+kk));

			job.waitForCompletion(true);
			bw2.write(loss_sum + "\n");
		}
		bw2.close();

    }
    
}
