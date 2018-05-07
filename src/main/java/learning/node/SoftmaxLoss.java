package learning.node;

import learning.tensor.Tensor;
import learning.optimize.Optimizer;
import learning.operation.Operation;

import org.jocl.*;
import static org.jocl.CL.*;
import static learning.ComputeContext.*;

import java.util.ArrayList;

public class SoftmaxLoss extends I2Node {
   
   private static cl_kernel[] kernels = new cl_kernel[5];
   
   private int[] row_size;
   
   //for derivative calculations
   private ArrayList<Tensor> softmax = new ArrayList<Tensor>(), labels = new ArrayList<Tensor>();
   
   private static double e = 0.00001;
   
   public SoftmaxLoss(Node n) {
      super();
      before2 = n;
   }
   
   public void forwardProp() {
      
      if(net.getTime() == softmax.size()) {
         softmax.add(Tensor.create(in.getSize()));
      } if(net.getTime() == labels.size()) {
         labels.add(Tensor.create(in2.getSize()));
      }
      
      Tensor soft = softmax.get(net.getTime()), labelB = labels.get(net.getTime());
      
      if(OPEN_CL) {
         copy(in2, labelB);
         
         CL.clSetKernelArg(kernels[0], 0, Sizeof.cl_mem, Pointer.to(in.getMem()));
         CL.clSetKernelArg(kernels[0], 1, Sizeof.cl_mem, Pointer.to(soft.getMem()));
         
         CL.clEnqueueNDRangeKernel(queue, kernels[0], 1, null, input.global_size, input.local_size, 0, null, null);
         
         CL.clSetKernelArg(kernels[1], 0, Sizeof.cl_mem, Pointer.to(soft.getMem()));
         CL.clSetKernelArg(kernels[1], 1, Sizeof.cl_mem, Pointer.to(output.getMem()));
         CL.clSetKernelArg(kernels[1], 2, Sizeof.cl_int, Pointer.to(row_size));
         
         CL.clEnqueueNDRangeKernel(queue, kernels[1], 1, null, output.global_size, output.local_size, 0, null, null);
         
         CL.clSetKernelArg(kernels[2], 0, Sizeof.cl_mem, Pointer.to(soft.getMem()));
         CL.clSetKernelArg(kernels[2], 1, Sizeof.cl_mem, Pointer.to(output.getMem()));
         CL.clSetKernelArg(kernels[2], 2, Sizeof.cl_int, Pointer.to(row_size));
         
         CL.clEnqueueNDRangeKernel(queue, kernels[2], 1, null, input.global_size, input.local_size, 0, null, null);
         
         
         //System.out.println(java.util.Arrays.toString(soft.unload()));
         
         CL.clSetKernelArg(kernels[3], 0, Sizeof.cl_mem, Pointer.to(soft.getMem()));
         CL.clSetKernelArg(kernels[3], 1, Sizeof.cl_mem, Pointer.to(in.getMem()));
         CL.clSetKernelArg(kernels[3], 2, Sizeof.cl_mem, Pointer.to(in2.getMem()));
         CL.clSetKernelArg(kernels[3], 3, Sizeof.cl_int, Pointer.to(row_size));
         
         CL.clEnqueueNDRangeKernel(queue, kernels[3], 1, null, input.global_size, input.local_size, 0, null, null);
         
         //System.out.println(java.util.Arrays.toString(in.unload()));
         
         CL.clSetKernelArg(kernels[1], 0, Sizeof.cl_mem, Pointer.to(in.getMem()));
         CL.clSetKernelArg(kernels[1], 1, Sizeof.cl_mem, Pointer.to(output.getMem()));
         CL.clSetKernelArg(kernels[1], 2, Sizeof.cl_int, Pointer.to(row_size));
         
         CL.clEnqueueNDRangeKernel(queue, kernels[1], 1, null, output.global_size, output.local_size, 0, null, null);
         
         //System.out.println(java.util.Arrays.toString(output.unload()));
         
      
      } else {
         
         //System.out.println(java.util.Arrays.toString(in.getVector(0)));
         
         for(int v = 0; v<in.vectorNum(); v++) {
            float[] vecO = soft.getVector(v), vecI = in.getVector(v);
            float sum = 0f, max = 0f;
            for(int i = 0; i<vecI.length; i++) {
               max = (float) Math.max(max,vecI[i]);
            }
            for(int i = 0; i<vecI.length; i++) {
               vecO[i] = (float) Math.exp(vecI[i] - max + e);
               sum += vecO[i];
            }
               sum+=e;
            for(int i = 0; i<vecI.length; i++) {
               vecO[i] = (vecO[i])/sum;
               if(!Float.isFinite(vecO[i])) {
                  System.out.println(java.util.Arrays.toString(in.getVector(0)));
               }
            }
         }
         
         //System.out.println(java.util.Arrays.toString(soft.getVector(0)));
         
         output.loadZeros();
         float[] vecL = in2.getVector(0), vecO = output.getVector(0);
         for(int v = 0; v<soft.vectorNum(); v++) {
            float[] vecS = soft.getVector(v);
            vecL = in2.getVector(v / vecL.length);
            vecO = output.getVector(v / vecO.length);
            for(int i = 0; i<vecS.length; i++) {
               if((int) (vecL[v]) == i) {
                  vecO[v] += (float) -Math.log(vecS[i]);
               } else {
                  vecO[v] += (float) -Math.log(1f-vecS[i]);
               }
            }
            vecO[v] /= vecS.length;
         }
         
         //System.out.println(java.util.Arrays.toString(output.getVector(0)));
         
         for(int v = 0; v<in2.vectorNum(); v++) {
            System.arraycopy(in2.getVector(v), 0, labelB.getVector(v), 0, labelB.getVector(v).length);
         }
         
      }
      
   }
   
   public void backwardProp() {
      
      Tensor labelB = labels.get(net.getTime()), soft = softmax.get(net.getTime());
      
      if(OPEN_CL) {
         
         //Calculate [softmax.get(time) - labels.get(time)] * dO => input
         
         CL.clSetKernelArg(kernels[4], 0, Sizeof.cl_mem, Pointer.to(labelB.getMem()));
         CL.clSetKernelArg(kernels[4], 1, Sizeof.cl_mem, Pointer.to(soft.getMem()));
         CL.clSetKernelArg(kernels[4], 2, Sizeof.cl_mem, Pointer.to(dO.getMem()));
         CL.clSetKernelArg(kernels[4], 3, Sizeof.cl_mem, Pointer.to(input.getMem()));
         CL.clSetKernelArg(kernels[4], 4, Sizeof.cl_int, Pointer.to(row_size));
         
         CL.clEnqueueNDRangeKernel(queue, kernels[4], 1, null, input.global_size, input.local_size, 0, null, null);
         
         
      } else {
         
         float[] vecL = labelB.getVector(0), vecdO = dO.getVector(0);
         for(int v = 0; v<input.vectorNum(); v++) {
            float[] vecdI = input.getVector(v), vecS = soft.getVector(v);
            vecL = labelB.getVector(v / vecL.length);
            vecdO = dO.getVector(v / vecdO.length);
            for(int i = 0; i<vecdI.length; i++) {
               if((int) (vecL[v]) == i) {
                  vecdI[i] = (vecS[i] - 1f) * vecdO[v];
               } else {
                  vecdI[i] = vecS[i] * vecdO[v];
               }
            }
            
         }
         
      }
      
   }
   
   void backwards(Tensor t) {
      dO = t;
      net.operate(this);
      before.backwards(input);
   }
   
   protected void attachBefore(Node b) {
      if(before == null) {
         before = b;
         outputSize = before2.getOutputSize();
         before2.chain(this);
         return;
      }
      
      prepare();
   }
   
   protected void prepare() {
      input = Tensor.create(before.getOutputSize());
      output = Tensor.create(outputSize);
      
      if(OPEN_CL) {
         if(!ready) {
            kernels[0] = getKernel("./kernel/softmax.c", "exponential");
            kernels[1] = getKernel("./kernel/reduce_rows.c", "reduce_rows");
            kernels[2] = getKernel("./kernel/divide.c", "divide");
            kernels[3] = getKernel("./kernel/crossentropy.c", "crossentropy");
            kernels[4] = getKernel("./kernel/dsoftloss.c", "dsoftloss");
            
            ready = true;
         }
         
         row_size = new int[] {(int) (input.global_size[0]/output.global_size[0])};
      }
   }
   
   public Tensor read() {
      return (Tensor) softmax.get(net.getTime());
   }
   
}