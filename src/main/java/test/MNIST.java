/*
Author : Augustin ANGAUD
*/

package test;

import learning.tensor.*;
import learning.node.*;
import learning.optimize.*;
import learning.*;

import org.jocl.*;
import static org.jocl.CL.*;
import learning.ComputeContext;
import static learning.ComputeContext.*;

import java.io.File;
import java.io.FileInputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;

public class MNIST {
   
   private static float[][] imagesD;
   private static float[] labelsD;
   
   public static boolean test() {
      
      if(!loadMNIST()) return false;
      
      ComputeContext.PROFILING = true;
      if(ComputeContext.OPEN_CL) {
         ComputeContext.init();
      }
      
      Network net = new Network();
      
      int[] iSize = {28*28}, hSize = {(int) pref_local_size*2}, oSize = {10}, lSize = {1};
      //hSize = new int[] {5};
      
      Input input = new Input(iSize), label = new Input(lSize);
      SoftmaxLoss softmax = (SoftmaxLoss) input.chain(new Linear(hSize)).chain(new Sigmoid()).chain(new Linear(oSize)).chain(new SoftmaxLoss(label));
      Output output = (Output) softmax.chain(new Output());
      
      Tensor inputs = Tensor.create(iSize), labels = Tensor.create(lSize);
      
      input.load(inputs);
      label.load(labels);
      
      //0.001, 0.9, 0.999, 0.00000001
      
      AdamOptimizer gd = new AdamOptimizer(0.001f, 0.9f, 0.999f, 0.00001f);
      //GradientDescent gd = new GradientDescent(0.0001f);
      long dt, load = 0L, unload = 0L, forward = 0L, backward = 0L, opt = 0L;
      double win = 0.0;
      Tensor loss = Tensor.create(lSize);
      boolean printloss = true, printwin = true, saveTensor = false;
      int numsteps = 100000;
      
      InputStreamReader fileInputStream = new InputStreamReader(System.in);
      BufferedReader bufferedReader = new BufferedReader(fileInputStream);
      
      clFinish(queue);
      
      System.out.println("Training");
      long time = System.currentTimeMillis();
      for(int t = 0; t<numsteps; t = (t+1)%imagesD.length) {
         
         inputs.load(imagesD[t]);
         labels.load(new float[] {labelsD[t]});
         
         net.calculate(output);
         
         if(printwin) {
            float[] soft = softmax.read().unload();
            int index = 0;
            for(int j = 1; j<soft.length; j++) {
               if(soft[j] > soft[index]) index = j;
            }
            if(index == labelsD[t]) win++;
         } if(printloss) {
            add(output.read(),loss);
         }
         
         int ts = 100;
         if(t%ts==ts-1) {
            
            if(printwin) {
               System.out.println(win);
               win = 0.0;
            } if(printloss) {
               System.out.println(loss.unload()[0]/ts);
               zeros(loss);
            }
            
         }
         
         net.gradientsFrom(softmax);
         
         
         net.optimize(softmax, gd);
         
         
         try {
            if(bufferedReader.ready()) {
               System.out.println("---------------------------------");
               System.out.println("q to quit");
               System.out.println("s to save");
               System.out.println("---------------------------------");
               String s = bufferedReader.readLine();
               if(s.equals("q")) {
                  numsteps = t;
                  break;
               } if(s.equals("s")) {
                  clFinish(queue);
                  Tensor in = Tensor.create(iSize);
                  in.loadOnes();
                  input.load(in);
                  float ce = 0f;
                  for(int i = 0; i<5000; i++) {
                     net.calculate(output);
                     float g = output.read().unload()[0];
                     ce+=g;
                     if(i%100 == 99) {
                        System.out.println(ce/100);
                        ce = 0f;
                        clFinish(queue);
                        saveTensor(in, labelsD[t]);
                     }
                     net.gradientsFrom(output);
                     gd.prepare();
                     gd.optimize(in, input.getGradient());
                     
                     
                  }
                  
                  in.dispose();
                  input.load(inputs);
               }
            }
         } catch(java.io.IOException e) {
            e.printStackTrace();
         }
         
         
      }
      
      clFinish(queue);
      
      System.out.println("Completed " + numsteps + " iterations of linear-sigmoid-linear-softmax-loss of hidden size " + hSize[0] + " in " + (float) (System.currentTimeMillis() - time) * 1000f / numsteps + " milliseconds per thousand step");
      System.out.println("Loading time      : " + load/(numsteps*1000));
      System.out.println("Forward time      : " + forward/(numsteps*1000));
      System.out.println("Unloading time    : " + unload/(numsteps*1000));
      System.out.println("Backward time     : " + backward/(numsteps*1000));
      System.out.println("Optimization time : " + opt/(numsteps*1000));
      
      try {
         bufferedReader.close();
      } catch(java.io.IOException e) {
         e.printStackTrace();
      }
      
      if(OPEN_CL) {
         ComputeContext.release();
      }
      
      return true;
   }
   
   private static void saveTensor(Tensor t, float label) {
      float[] array = t.unload();
      try {
         PrintWriter out = new PrintWriter("./data/input-gradients.txt");
         for(int i = 0; i<array.length; i++) {
            out.print(array[i] + ",");
         }
         out.print(label);
         out.close();
      } catch(IOException e) {
         e.printStackTrace();
      }
   }
   
   private static boolean loadMNIST() {
      
      File file = new File("./data/train-labels-idx1-ubyte"), file2 = new File("./data/train-images-idx3-ubyte");
      FileInputStream fin = null, fin2 = null;
      byte[] labelsBytes, imagesBytes;
      try {
         // create FileInputStream object
         fin = new FileInputStream(file);
         labelsBytes = new byte[(int) file.length()];
          
         // Reads up to certain bytes of data from this input stream into an array of bytes.
         fin.read(labelsBytes);
         
         fin2 = new FileInputStream(file2);
         imagesBytes = new byte[(int) file2.length()];
          
         // Reads up to certain bytes of data from this input stream into an array of bytes.
         fin2.read(imagesBytes);
      } catch (FileNotFoundException e) {
         System.out.println("File not found" + e);
         return false;
      } catch (IOException ioe) {
         System.out.println("Exception while reading file " + ioe);
         return false;
      } finally {
         try {
            if (fin != null) {
               fin.close();
            } if(fin2 != null) {
               fin2.close();
            }
         } catch (IOException ioe) {
             System.out.println("Error while closing stream: " + ioe);
         }
      }
      
      if(labelsBytes != null && imagesBytes != null) {
         
         int numElements = 0;
         for(int i = 0; i<4; i++) {
            numElements += (labelsBytes[i+4] & 0xFF) * Math.pow(256,3-i);
         }
         
         labelsD = new float[numElements];
         
         for(int i = 0; i<numElements; i++) {
            labelsD[i] = labelsBytes[i+8] & 0xFF;
         }
         
         numElements = 0;
         for(int i = 0; i<4; i++) {
            numElements += (imagesBytes[i+4] & 0xFF) * Math.pow(256,3-i);
         }
         
         int nrow = 0;
         for(int i = 0; i<4; i++) {
            nrow += (imagesBytes[i+8] & 0xFF) * Math.pow(256,3-i);
         }
         
         int ncol = 0;
         for(int i = 0; i<4; i++) {
            ncol += (imagesBytes[i+12] & 0xFF) * Math.pow(256,3-i);
         }
         
         imagesD = new float[numElements][nrow*ncol];
         
         for(int i = 0; i<numElements; i++) {
            for(int j = 0; j<nrow*ncol; j++) {
               imagesD[i][j] = (imagesBytes[i*nrow*ncol+j+16] & 0xFF)/256f;
            }
         }
         
         System.out.println("MNIST Dataset loaded successfully");
         return true;
      }
      
      return false;
   }
   
}
