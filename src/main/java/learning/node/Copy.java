package learning.node;

import learning.tensor.Tensor;
import learning.optimize.Optimizer;
import learning.operation.Operation;

import java.util.Arrays;

import org.jocl.*;
import static org.jocl.CL.*;

import static learning.ComputeContext.*;

public class Copy extends O2Node {
   
   private cl_kernel kernel;
   
   void forwardProp() {
      output = in;
   }
   
   void backwardProp() {
      
      if(OPEN_CL) {
         CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(dO.getMem()));
         CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(dO2.getMem()));
         CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(input.getMem()));
         
         CL.clEnqueueNDRangeKernel(queue, kernel, 1, null, input.global_size, input.local_size, 0, null, null);
      } else {
         
      }
   }
   
   Tensor forwards(Operation op) {
      
      return output;
   }
   
   void backwards(Tensor t, Operation op) {
      
   }
   
   protected void prepare() {
      input = Tensor.create(outputSize);
      
      if(OPEN_CL) {
         if(!ready) {
            kernel = getKernel("./kernel/pointmul.c", "pointmul");
            ready = true;
         }
      }
   }
   
}