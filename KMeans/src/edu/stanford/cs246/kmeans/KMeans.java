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
		return (float)sqrt(dist);
	}
	public static float DistM(float[] a, float[] b) {
		float dist = 0;
		for (int i = 0; i < a.length; i++) {
			dist += abs((double)(a[i] - b[i]));
		}
		return dist;
	}
	
	public static class Map extends
			Mapper<LongWritable, Text, Text, ArrayWritable> {
		
		private Text cluster = new Text();
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			ArrayWritable point_wr = new ArrayWritable(FloatWritable.class);
			Configuration conf = context.getConfiguration();
			String cent_loc = conf.get("cent_loc");
			float[][] center = ReadCenter(cent_loc);
			//System.out.println(value);
			String[] parts = value.toString().split(" ");
			float[] point = new float[parts.length];
            for (int j = 0; j < parts.length; j++) {
            	point[j] = Float.parseFloat(parts[j]);
            }
            // compute distance with each center and assign cluster
            float dist;
            float dist_min;
            int loc = 0;
            for (int i = 0; i < 10; i++) {
            	dist = DistE(point, center[i]);
            	if (i == 0) {
            		dist_min = dist;
            	}
            	if (dist < dist_min) {
            		loc = i;
            		dist_min = dist;
            	}
            }
            cluster.set(""+loc);
            point_wr.set(point);
            context.write(cluster, point_wr);
		}
	}
	
 
    public static class Reduce extends Reducer<Text, Center, Text, float[]> {
        public void reduce(Text key, Iterable<float[]> values, Context context) throws IOException, InterruptedException {
        	float[] center = new float[10];
        	for ( int i = 0; i < 10; i++) {center[i] = 0;}
        	int count = 0;
            for (float[] val : values) {
               for (int i = 0; i < 10; i++) {
            	   center[i] += val[i];
               }
               count += 1;
            }
            for (int i = 0; i < 10; i++ ) {center[i] = center[i] / count;}
            context.write(key, center);
            
        }
    }
    

	
	public static float[][] ReadCenter(String path) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(path));        
        float[][] c1 = new float[10][58];
        for (int i = 0; i < 10; i++) {
        	String line = br.readLine();
        	String[] parts = line.split(" ");
            for (int j = 0; j < parts.length; j++) {
            	c1[i][j] = Float.parseFloat(parts[j]);
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

		// System.out.println(c2[9][54]);
        
		for (int kk = 0; kk < 1; kk++) {
			Job job = new Job(conf, "job");
			job.setJarByClass(KMeans.class);
			job.setOutputKeyClass(Text.class);
			job.setOutputValueClass(IntWritable.class);

			job.setMapperClass(Map.class);
			job.setReducerClass(Reduce.class);

			job.setInputFormatClass(TextInputFormat.class);
			job.setOutputFormatClass(TextOutputFormat.class);
			job.setMapOutputValueClass(ArrayWritable.class);
			
			
			FileInputFormat.addInputPath(job, new Path("data.txt"));
			if (kk == 0) {
				conf.set("cent_loc", "c1.txt");
			}
			else {conf.set("output/cent_loc", "c1_"+kk+".txt");}
			
			FileOutputFormat.setOutputPath(job, new Path("output"));

			job.waitForCompletion(true);
		}


    }
    
}
