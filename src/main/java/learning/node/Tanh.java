package learning.node;

import learning.tensor.Tensor;

import org.jocl.*;

import static learning.ComputeContext.*;

import java.util.ArrayList;

public class Tanh extends Node {
   
   //for derivative calculations
   private final ArrayList<Tensor> outputs = new ArrayList<Tensor>();
   
   private static cl_kernel kernel, kernel2;
   
   @Override
   public void forwardProp() {
      if(time == outputs.size()) {
         outputs.add(Tensor.create(before.getOutputSize()));
      }
      
      if(OPEN_CL) {
         CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(in.getMem()));
         CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(output.getMem()));
         CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(outputs.get(time).getMem()));
         
         CL.clEnqueueNDRangeKernel(queue, kernel, 1, null, output.global_size, output.local_size, 0, null, null);
      } else {
         for(int v = 0; v<in.vectorNum(); v++) {
            float[] vecO = output.getVector(v), vecI = in.getVector(v), vecT = outputs.get(time).getVector(v);
            for(int i = 0; i<vecI.length; i++) {
               double expp = Math.exp(vecI[i]), expm = Math.exp(-vecI[i]);
               vecO[i] = (float) ((expp - expm)/(expp + expm));
               vecT[i] = vecO[i];
            }
         }
      }
      
      time++;
   }
   
   @Override
   public void backwardProp() {
      time--;
      
      if(OPEN_CL) {
         
         CL.clSetKernelArg(kernel2, 0, Sizeof.cl_mem, Pointer.to(dO.getMem()));
         CL.clSetKernelArg(kernel2, 1, Sizeof.cl_mem, Pointer.to(outputs.get(time).getMem()));
         CL.clSetKernelArg(kernel2, 2, Sizeof.cl_mem, Pointer.to(input.getMem()));
         
         CL.clEnqueueNDRangeKernel(queue, kernel2, 1, null, input.global_size, input.local_size, 0, null, null);
         
      } else {
         for(int v = 0; v<dO.vectorNum(); v++) {
            float[] vecdI = input.getVector(v), vecdO = dO.getVector(v), vecT = outputs.get(time).getVector(v);
            for(int i = 0; i<vecdO.length; i++) {
               vecdI[i] = (1 - vecT[i] * vecT[i]) * vecdO[i];
            }
         }
      }
   }
   
   @Override
   protected void prepare() {
      output = Tensor.create(outputSize);
      input = output;
      
      if(OPEN_CL) {
         if(!ready) {
            kernel = getKernel("./kernel/tanh.c", "tanh_");
            kernel2 = getKernel("./kernel/dtanh.c", "dtanh");
            ready = true;
         }
      }
   }
   
}