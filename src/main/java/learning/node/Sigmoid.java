package learning.node;

import learning.tensor.Tensor;
import learning.optimize.Optimizer;
import learning.operation.Operation;

import org.jocl.*;
import static org.jocl.CL.*;

import static learning.ComputeContext.*;

import java.util.ArrayList;

public class Sigmoid extends Node {
   
   //for derivative calculations
   private ArrayList<Tensor> outputs = new ArrayList<Tensor>();
   
   private static cl_kernel kernel, kernel2;
   
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
               vecO[i] = (float) (1.0 / (1.0 + Math.exp(-vecI[i])));
               vecT[i] = vecO[i];
            }
         }
      }
      
      
      time++;
   }
   
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
               vecdI[i] = vecT[i] * (1f - vecT[i]) * vecdO[i];
            }
         }
      }
   }
   
   protected void prepare() {
      output = Tensor.create(outputSize);
      input = output;
      
      if(OPEN_CL) {
         if(!ready) {
            kernel = getKernel("./kernel/sigmoid.c", "sigmoid");
            kernel2 = getKernel("./kernel/dsigmoid.c", "dsigmoid");
            ready = true;
         }
      }
   }
   
}