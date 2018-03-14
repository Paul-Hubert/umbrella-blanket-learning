package learning.node;

import learning.tensor.Tensor;
import learning.optimize.Optimizer;
import learning.operation.Operation;

import org.jocl.*;
import static org.jocl.CL.*;

import static learning.ComputeContext.*;

public class PointAdd extends I2Node {
   
   private cl_kernel kernel;
   
   void forwardProp() {
      
      if(OPEN_CL) {
         CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(in.getMem()));
         CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(in2.getMem()));
         CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(output.getMem()));
         
         CL.clEnqueueNDRangeKernel(queue, kernel, 1, null, output.global_size, output.local_size, 0, null, null);
      } else {
         for(int v = 0; v<in.vectorNum(); v++) {
            float[] vecI1 = in.getVector(v), vecI2 = in2.getVector(v), vecO = output.getVector(v);
            for(int i = 0; i<vecO.length; i++) {
               vecO[i] = vecI1[i] + vecI2[i];
            }
         }
      }
   }
   
   void backwardProp() {
      input = dO;
      input2 = dO;
   }
   
   protected void prepare() {
      output = Tensor.create(outputSize);
      input = Tensor.create(outputSize);
      input2 = Tensor.create(outputSize);
      
      if(OPEN_CL) {
         if(!ready) {
            kernel = getKernel("./kernel/pointadd.c", "pointadd");
            ready = true;
         }
      }
   }
   
}