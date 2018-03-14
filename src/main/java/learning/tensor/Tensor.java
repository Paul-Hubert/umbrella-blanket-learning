package learning.tensor;

import org.jocl.*;
import static org.jocl.CL.*;

import static learning.ComputeContext.*;
import learning.ComputeContext;

import java.util.ArrayList;

public class Tensor {
   
   private final static ArrayList<cl_mem> memory = new ArrayList<cl_mem>();
   
   protected int[] size;
   protected int bufferSize;
   public final long[] global_size, local_size;
   protected boolean rowMajor = true;
   
   private cl_mem data;
   
   public boolean isZero = false;
   
   public Tensor (int[] size) {
      this.size = size;
      bufferSize = 1;
      for(int i = 0; i < size.length; i++) {
         bufferSize *= size[i];
      }
      global_size = new long[] {bufferSize};
      local_size = new long[] {(global_size[0]%pref_local_size==0) ? pref_local_size : 1};
   }
   
   public Tensor (Tensor f, int[] newSize) {
      bufferSize = f.getBufferSize();
      this.size = newSize;
      global_size = f.global_size;
      local_size = f.local_size;
      data = f.getMem();
   }
   
   public static Tensor create(int[] size) {
      if(OPEN_CL) {
         return new Tensor(size).initCL();
      } else {
         if(size.length == 0) {
            return new Tensor1(new int[] {1});
         } if(size.length == 1) {
            return new Tensor1(size);
         } if(size.length == 2) {
            return new Tensor2(size);
         } if(size.length == 3) {
            return new Tensor3(size);
         }
         System.out.println("incorrect tensor size");
         return null;
      }
   }
   
   public Tensor initCL() {
      data = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_float * bufferSize, null, null);
      ComputeContext.zeros(this);
      memory.add(data);
      return this;
   }
   
   public void load(float[] val) {
      if(OPEN_CL) {
         if(val.length != bufferSize) System.out.println("Incorrect argument value length");
         clEnqueueWriteBuffer(queue, data, CL_TRUE, 0, Sizeof.cl_float * bufferSize, Pointer.to(val), 0, null, null);
      }
   }
   
   public float[] unload() {
      if(OPEN_CL) {
         float[] dest = new float[bufferSize];
         // Read the output data
         clEnqueueReadBuffer(queue, data, CL_TRUE, 0, Sizeof.cl_float * bufferSize, Pointer.to(dest), 0, null, null);
         return dest;
      } else {
         return getVector(0);
      }
   }
   
   public float[] zeros() {
      float[] vec = new float[bufferSize];
      for(int i = 0; i<bufferSize; i++) {
         vec[i] = 0f;
      }
      return vec;
   }
   
   public float[] ones() {
      float[] vec = new float[bufferSize];
      for(int i = 0; i<bufferSize; i++) {
         vec[i] = 1f;
      }
      return vec;
   }
   
   public float[] random(float m, float M) {
      float[] vec = new float[bufferSize];
      for(int i = 0; i<bufferSize; i++) {
         vec[i] = (float) (Math.random()*(M-m)+m);
      }
      return vec;
   }
   
   public Tensor loadZeros() {
      if(OPEN_CL) {
         ComputeContext.zeros(this);
      } else {
         for(int v = 0; v<vectorNum(); v++) {
            float[] vec = getVector(v);
            for(int i = 0; i<vec.length; i++) {
               vec[i] = 0f;
            }
         }
      }
      return this;
   }
   
   public Tensor loadOnes() {
      if(OPEN_CL) {
         ComputeContext.ones(this);
      } else {
         for(int v = 0; v<vectorNum(); v++) {
            float[] vec = getVector(v);
            for(int i = 0; i<vec.length; i++) {
               vec[i] = 1f;
            }
         }
      }
      return this;
   }
   
   public Tensor loadRandom(float m, float M) {
      if(OPEN_CL) {
         this.load(this.random(m, M));
      } else {
         for(int v = 0; v<vectorNum(); v++) {
            float[] vec = getVector(v);
            for(int i = 0; i<vec.length; i++) {
               vec[i] = (float) (Math.random()*(M-m)+m);
            }
         }
      }
      return this;
   }
   
   public float[] getVector(int i) {
      return null;
   }
   public int vectorNum() {
      return 0;
   }
   public float[][] getMatrix(int i) {
      return null;
   }
   public int matrixNum() {
      return 0;
   }
   
   public int[] getSize() {
      return size;
   }
   
   public int getBufferSize() {
      return bufferSize;
   }
   
   public cl_mem getMem() {
      return data;
   }
   
   public long[] global_size() {
      return global_size;
   }
   
   public long[] local_size() {
      return local_size;
   }
   
   public void dispose() {
      CL.clReleaseMemObject(data);
      memory.remove(data);
   }
   
   public static void release() {
      for(cl_mem mem : memory) {
         CL.clReleaseMemObject(mem);
      }
   }
   
}