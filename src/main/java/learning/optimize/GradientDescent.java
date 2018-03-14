package learning.optimize;

import learning.tensor.*;

import org.jocl.*;
import static org.jocl.CL.*;
import static learning.ComputeContext.*;

public class GradientDescent extends Optimizer {
   
   float learn_rate;
   float[] _learn_rate;
   
   private cl_kernel kernel;
   
   public GradientDescent (float lr) {
      learn_rate = lr;
      _learn_rate = new float[] {lr};
      if(OPEN_CL) {
         kernel = getKernel("./kernel/sgd.c", "sgd");
      }
   }
   
   public void prepare() {
      
   }
   
   public void optimize(Tensor param, Tensor dparam) {
      //assumes that param and dparam are same size
      if(OPEN_CL) {
         
         CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(param.getMem()));
         CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(dparam.getMem()));
         CL.clSetKernelArg(kernel, 2, Sizeof.cl_float, Pointer.to(_learn_rate));
         
         CL.clEnqueueNDRangeKernel(queue, kernel, 1, null, param.global_size, param.local_size, 0, null, null);
         
      } else {
         for(int v = 0; v < param.vectorNum(); v++) {
            float[] vecP = param.getVector(v), vecdP = dparam.getVector(v);
            for(int i = 0; i<vecP.length; i++) {
               vecP[i] -= learn_rate*vecdP[i];
               vecdP[i] = 0f;
            }
         }
      }
      
   }
   
}