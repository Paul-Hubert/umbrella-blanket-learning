package learning.node;

import learning.tensor.*;
import learning.optimize.*;

import org.jocl.*;
import static org.jocl.CL.*;

import java.util.ArrayList;

import learning.tensor.Tensor;
import learning.optimize.Optimizer;
import learning.operation.Operation;

import static learning.ComputeContext.*;

public class Linear extends Node {
   
   //for derivative calculations
   private ArrayList<Tensor> inputs = new ArrayList<Tensor>();
   
   private Tensor weights, dweights, bias, dbias;
   private Tensor work;
   
   private static cl_kernel kernel, kernel2;
   
   private int[] ncol,mrow;
   
   public Linear(int[] size) {
      outputSize = size;
   }
   
   void forwardProp() {
      
      if(time == inputs.size()) {
         inputs.add(Tensor.create(input.getSize()));
      }
      
      if(OPEN_CL) {
         copy(in, inputs.get(time));
         
         CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(weights.getMem()));
         CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(in.getMem()));
         CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(bias.getMem()));
         CL.clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(output.getMem()));
         CL.clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(ncol));
         
         CL.clEnqueueNDRangeKernel(queue, kernel, 1, null, output.global_size, output.local_size, 0, null, null);
         
         //System.out.println(java.util.Arrays.toString(in.unload()));
         
      } else {
         output.loadZeros();
         float[] vecB = bias.getVector(0);
         float[][] matW = weights.getMatrix(0);
         for(int v = 0; v < output.vectorNum(); v++) {
            float[] vecI = in.getVector(v), vecO = output.getVector(v), vecT = ((Tensor) inputs.get(time)).getVector(v);
            System.arraycopy(vecI, 0, vecT, 0, vecT.length);
            for(int i = 0; i<mrow[0]; i++) {
               for(int j = 0; j<ncol[0]; j++) {
                  vecO[i] += vecI[j]*matW[j][i];
               }
               vecO[i] += vecB[i];
            }
         }
      }
      
      // output = input * weights + bias
      
      time++;
   }
   
   void backwardProp() {
      time--;
      Tensor inputB = inputs.get(time);
      
      if(OPEN_CL) {
         
         CL.clSetKernelArg(kernel2, 0, Sizeof.cl_mem, Pointer.to(weights.getMem()));
         CL.clSetKernelArg(kernel2, 1, Sizeof.cl_mem, Pointer.to(dweights.getMem()));
         CL.clSetKernelArg(kernel2, 2, Sizeof.cl_mem, Pointer.to(dbias.getMem()));
         CL.clSetKernelArg(kernel2, 3, Sizeof.cl_mem, Pointer.to(inputB.getMem()));
         CL.clSetKernelArg(kernel2, 4, Sizeof.cl_mem, Pointer.to(dO.getMem()));
         CL.clSetKernelArg(kernel2, 5, Sizeof.cl_mem, Pointer.to(input.getMem()));
         CL.clSetKernelArg(kernel2, 6, Sizeof.cl_int, Pointer.to(mrow));
         
         CL.clEnqueueNDRangeKernel(queue, kernel2, 1, null, input.global_size, input.local_size, 0, null, null);
         
         add(dO,dbias);
         
      } else {
         input.loadZeros();
         float[] vecdB = dbias.getVector(0);
         float[][] matW = weights.getMatrix(0), matdW = dweights.getMatrix(0);
         for(int v = 0; v < input.vectorNum(); v++) {
            float[] vecdI = input.getVector(v), vecdO = dO.getVector(v), vecT = inputB.getVector(v);
            for(int i = 0; i<vecdO.length; i++) {
               for(int j = 0; j<vecdI.length; j++) {
                  vecdI[j] += matW[j][i]*vecdO[i];
                  matdW[j][i] += vecT[j]*vecdO[i] / input.vectorNum();
               }
               
               vecdB[i] += vecdO[i] / input.vectorNum();
            }
         }
      }
      
   }
   
   public void optimize(Optimizer opt) {
      
      opt.optimize(weights,dweights);
      opt.optimize(bias,dbias);
      
   }
   
   protected void setOutputSize() {}
   
   protected void prepare() {
      int[] inputSize = before.getOutputSize();
      int[] weightSize = {inputSize[inputSize.length-1], outputSize[outputSize.length-1]}, biasSize = {outputSize[outputSize.length-1]};
      
      weights = Tensor.create(weightSize).loadRandom(-0.1f,0.1f);
      dweights = Tensor.create(weightSize).loadZeros();
      bias = Tensor.create(biasSize).loadRandom(-0.1f,0.1f);
      dbias = Tensor.create(biasSize).loadZeros();
      
      input = Tensor.create(inputSize);
      output = Tensor.create(outputSize);
      
      if(OPEN_CL) {
         if(!ready) {
            kernel = getKernel("./kernel/gemv3.c", "gemv1");
            kernel2 = getKernel("./kernel/dlinear.c", "dlinear");
            ready = true;
         }
         
         ncol = new int[] {input.getSize()[0]};
         mrow = new int[] {output.getSize()[0]};
         
      }
   }
   
}