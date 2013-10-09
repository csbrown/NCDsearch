/* 
   I hereby release this work into the public domain, to the extent allowed by law, per the CC0 1.0 license 
   With Love,
   -Scott!
   gitpushoriginmaster@gmail.com
*/


package NCDSearch;

import java.io.*;
import java.util.*;
import java.util.zip.Deflater;
import java.net.*;

import org.apache.commons.io.FileUtils;

import org.apache.hadoop.fs.*;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;
import org.apache.hadoop.filecache.*;

//Usage: $ hadoop jar thenameofthisjar.jar <searchedforfile> <searchedindirectory> <output>
//TODO: Wrap this into a friendly printed message


/*
 * This is a tool to search for a file in a directory by its contents.  You provide some bytes and a directory to search, and it tells you
 * which files have hunks of your bytes in them according to a score called the normalized compression distance.  The score is from 0 to 1ish,
 * where a score close to 0 indicates that a file contains the searched for bytes almost identically, and a score close to 1 indicates that a
 * file does not contain anything even remotely like your bytes.  Works best for files sized at least 10 kilobytes or so.
 * TODO: The target file is read entirely into memory so has a maximum size limit.
*/
public class DistributedNCDSearch extends Configured implements Tool {

	//Constants
	private static final int COMPRESSION = 2;

	public static class ChunkyFileInputFormat extends FileInputFormat<Path, BytesWritable> {

		@Override
		public RecordReader<Path, BytesWritable> getRecordReader(InputSplit split, JobConf job, Reporter reporter) throws IOException {
            		return new ChunkyFileRecordReader((FileSplit) split, job);   
		}
	}

	/*
 	* Basically the standard RecordReader.  Reads the file with the standard filesplit into a bytestream and
	* keeps track of the name/location of the file
	*/
	public static class ChunkyFileRecordReader implements RecordReader<Path, BytesWritable> {

 		private FileSplit fileSplit;
		private Configuration conf;
		private boolean processed = false;

		public ChunkyFileRecordReader(FileSplit fileSplit, Configuration conf) throws IOException {
			this.fileSplit = fileSplit;
			this.conf = conf;
		}

		@Override
		public Path createKey() {
			return fileSplit.getPath();
		}

		@Override
		public BytesWritable createValue() {
			return new BytesWritable();
		}

		@Override
		public long getPos() throws IOException {
			return processed ? fileSplit.getLength() : 0;
		}

		@Override
		public float getProgress() throws IOException {
			return processed ? 1.0f : 0.0f;
		}

		@Override
		//The key is already read with the getKey() method.  This puts the fileSplit into a BytesWritable so that we can compress it later.
		public boolean next(Path key, BytesWritable value) throws IOException {
			if (!processed) {
				byte[] contents = new byte[(int) fileSplit.getLength()];
				Path file = fileSplit.getPath();
				FileSystem fs = file.getFileSystem(conf);
				FSDataInputStream in = null;
				try {
					in = fs.open(file);
					IOUtils.readFully(in, contents, (int)fileSplit.getStart(), (int)fileSplit.getLength());
					value.setSize((int)fileSplit.getLength());
					value.set(contents, 0, contents.length);
				} finally {
					IOUtils.closeStream(in);
				}
				processed = true;
				return true;
			}
			return false;
		}

		@Override
		public void close() throws IOException {
		//do nothing
		}
	}

	/*
 	 * Computes a modified one directional normalized compression distance.  Computes a small distance if the target is a subfile of file. 
 	 * This works by compressing each file, concatenating them and compressing the result, and then comparing the two files compressed 
 	 * separately vs. the two files compressed together.  If they compress relatively a lot when concatenated, then one is probably part
 	 * of the other.
 	 */
	public static float NCD(byte[] file, byte[] target, int compression) {

		Deflater compressor = new Deflater(compression);
		
		//This is where we dump our compressed bytes.  All we need is the size of this thing.  
    //TODO: In theory the compressed bytes could exceed the length of the target files...
		byte[] outputtrash = new byte[file.length + target.length];

		int bothcompressedsize;
		int filecompressedsize;
		int targetcompressedsize;

		//puts the target file and the searched file together.
		byte[] both = new byte[file.length + target.length];
		for (int i = 0; i < file.length; i++) {
			both[i] = file[i];
		}
		for (int i = 0; i < target.length; i++) {
			both[i + file.length] = target[i];
		}

		compressor.setInput(file); compressor.finish(); filecompressedsize = compressor.deflate(outputtrash); compressor.reset();
		compressor.setInput(target); compressor.finish(); targetcompressedsize = compressor.deflate(outputtrash); compressor.reset();
		compressor.setInput(both) ; compressor.finish(); bothcompressedsize  = compressor.deflate(outputtrash); compressor.reset();

		return (float)(bothcompressedsize - filecompressedsize)/(float)targetcompressedsize;
	}	

	public static class Map extends MapReduceBase implements Mapper<Path, BytesWritable, Text, FloatWritable> {

		private byte[] targetbytes;
		private Path target;

		//gets a copy of the target file out of our distributed cache
		public void configure(JobConf job) {
                        try {
				this.target = DistributedCache.getLocalCacheFiles(job)[0];
			} catch (IOException e) {
				//TODO: Make this do something.
			}
		}

		/* 
 		 * takes a <Path, Bytes> and returns <String, Score> where the String is just the human-readable version of the passed
		 * in key, and the Score is the NCD between our target bytes and the passed in Bytes.
		 */
		public void map(Path key, BytesWritable value, OutputCollector<Text, FloatWritable> output, Reporter reporter) throws IOException {
			this.targetbytes = FileUtils.readFileToByteArray(new File(target.toString()));
			float score;
			if (value.getLength() > 0) {
				byte[] thesebytes = value.getBytes();
				score = NCD(thesebytes, targetbytes, COMPRESSION);
				output.collect(new Text(key.toString()), new FloatWritable(score));
			}
		}
	}

	public static class Reduce extends MapReduceBase implements Reducer<Text, FloatWritable, Text, FloatWritable> {
		/*
 		 * takes a <String, Score> that is a file path and NCD score from the mapper, and computes the minimum for each key.
 		 * Basically, since we're comparing our target bytes with possibly only a chunk of larger files, any split up files
 		 * will have multiple entries corresponding to the distance from our target bytes to that particular chunk.  Here, we
 		 * associate each file with the smallest of those scores.
 		 */ 				
		public void reduce(Text key, Iterator<FloatWritable> values, OutputCollector<Text, FloatWritable> output, Reporter reporter) 
		throws IOException {
			float current;
			//Arbitrary large number.  NCDs are never much bigger than 1, so this is bigger than any possible NCD.
			float min = 1000000;
			while(values.hasNext()) {
				current = values.next().get();
				min = min < current ? min : current;
			} 
			output.collect(key, new FloatWritable(min));
		}
	}

	public int run(String args[]) throws Exception {

		String inputpath = args[1];
		String outputpath = args[2];

		JobConf conf = new JobConf(getConf(), ChunkyFileInputFormat.class);
		
		//Add the target file to a cache so all nodes can have a copy.
		DistributedCache.addCacheFile(new URI(args[0]), conf);		
	
		FileOutputFormat.setOutputPath(conf, new Path(outputpath));
		FileInputFormat.setInputPaths(conf, new Path(inputpath));
		
		conf.setJobName("NCDSearch");
		conf.setOutputKeyClass(Text.class);
		conf.setOutputValueClass(FloatWritable.class);
		conf.setOutputFormat(TextOutputFormat.class);
	
		conf.setInputFormat(ChunkyFileInputFormat.class);
		conf.setMapperClass(Map.class);
		conf.setReducerClass(Reduce.class);

		JobClient.runJob(conf);
		
		return 0;
	}

	public static void main(String args[]) throws Exception {

		int res = ToolRunner.run(new DistributedNCDSearch(), args);
		System.exit(res);
	}
			
}	
			
			
			
				
