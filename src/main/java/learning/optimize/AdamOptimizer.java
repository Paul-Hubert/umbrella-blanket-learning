package learning.optimize;

import learning.tensor.*;

import org.jocl.*;
import static org.jocl.CL.*;
import static learning.ComputeContext.*;

import java.util.HashMap;

public class AdamOptimizer extends Optimizer {
   
   /*
   learning_rate=0.001,
   beta1=0.9,
   beta2=0.999,
   epsilon=1e-08,
   
   t <- t + 1
   lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
   
   m_t <- beta1 * m_{t-1} + (1 - beta1) * g
   v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
   variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
   */
   
   private final float learning_rate, beta1, beta2, epsilon;
   private float t = 0f, lr_t = 0f;
   private Tensor t_;
   
   private cl_kernel kernelprep, kernel;
   
   private HashMap var;
   
   public AdamOptimizer (float learn_rate, float beta1_, float beta2_, float epsilon_) {
      learning_rate = learn_rate;
      beta1 = beta1_;
      beta2 = beta2_;
      epsilon = epsilon_;
      var = new HashMap();
      
      t_ = Tensor.create(new int[] {6});
      t_.load(new float[] {t, lr_t, learning_rate, beta1, beta2, epsilon});
      if(OPEN_CL) {
         kernelprep = getKernel("./kernel/adamprep.c", "adamprep");
         kernel = getKernel("./kernel/adam.c", "adam");
      }
   }
   
   public void prepare() {
      if(OPEN_CL) {
         CL.clSetKernelArg(kernelprep, 0, Sizeof.cl_mem, Pointer.to(t_.getMem()));
         
         long[] g = {1L};
         CL.clEnqueueNDRangeKernel(queue, kernelprep, 1, null, g, g, 0, null, null);
         
      } else {
         t++;
         lr_t = (float) (learning_rate * Math.sqrt(1.0 - Math.pow(beta2,t)) / (1.0 - Math.pow(beta1,t)));
      }
      
   }
   
   public void optimize(Tensor param, Tensor dparam) {
      
      if(!var.containsKey(param)) {
         Tensor[] dt = new Tensor[2];
         dt[0] = Tensor.create(param.getSize());
         dt[1] = Tensor.create(param.getSize());
         dt[0].loadZeros();
         dt[1].loadZeros();
         var.put(param, dt);
      }
      
      Tensor[] m_v = (Tensor[]) var.get(param);
      
      if(OPEN_CL) {
         
         CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(param.getMem()));
         CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(dparam.getMem()));
         CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(m_v[0].getMem()));
         CL.clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(m_v[1].getMem()));
         CL.clSetKernelArg(kernel, 4, Sizeof.cl_mem, Pointer.to(t_.getMem()));
         
         CL.clEnqueueNDRangeKernel(queue, kernel, 1, null, param.global_size, param.local_size, 0, null, null);
         
         //System.out.println(java.util.Arrays.toString(m_v[0].unload()));
         
      } else {
         for(int v = 0; v < param.vectorNum(); v++) {
            float[] vecP = param.getVector(v), vecdP = dparam.getVector(v), vecM = m_v[0].getVector(v), vecV = m_v[1].getVector(v);
            for(int i = 0; i<vecdP.length; i++) {
               vecM[i] = (float) (beta1 * vecM[i] + (1.0 - beta1) * vecdP[i]);
               vecV[i] = (float) (beta2 * vecV[i] + (1.0 - beta2) * vecdP[i] * vecdP[i]);
               vecP[i] -= (float) (lr_t * vecM[i] / (Math.sqrt(vecV[i]) + epsilon));
               vecdP[i] = 0f;
            }
         }
      }
      
   }
   
}