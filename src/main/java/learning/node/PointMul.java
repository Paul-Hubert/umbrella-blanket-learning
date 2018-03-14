package learning.node;

import learning.tensor.Tensor;
import learning.optimize.Optimizer;
import learning.operation.Operation;

import org.jocl.*;
import static org.jocl.CL.*;

import static learning.ComputeContext.*;

import java.util.ArrayList;

public class PointMul extends I2Node {
   
   private cl_kernel kernel;
   
   //for derivative calculations
   private ArrayList<Tensor> inputs1 = new ArrayList<Tensor>(), inputs2 = new ArrayList<Tensor>();
   
   void forwardProp() {
      
      if(time == inputs1.size()) {
         inputs1.add(Tensor.create(input.getSize()));
         inputs2.add(Tensor.create(input2.getSize()));
      }
      
      if(OPEN_CL) {
         CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(in.getMem()));
         CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(in2.getMem()));
         CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(output.getMem()));
         
         CL.clEnqueueNDRangeKernel(queue, kernel, 1, null, output.global_size, output.local_size, 0, null, null);
         
         copy(in, inputs1.get(time));
         copy(in2, inputs2.get(time));
         
      } else {
         for(int v = 0; v<in.vectorNum(); v++) {
            float[] vecI1 = in.getVector(v), vecI2 = in2.getVector(v), vecO = output.getVector(v);
            for(int i = 0; i<vecO.length; i++) {
               vecO[i] = vecI1[i] * vecI2[i];
            }
         }
      }
      
      //output = input * input2
      //inputs1.set(time, before.data)
      //inputs2.set(time, before2.data)
      
      time++;
   }
   
   void backwardProp() {
      time--;
      
      if(OPEN_CL) {
         CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(dO.getMem()));
         CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(inputs1.get(time).getMem()));
         CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(input.getMem()));
         
         CL.clEnqueueNDRangeKernel(queue, kernel, 1, null, input.global_size, input.local_size, 0, null, null);
         
         CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(dO.getMem()));
         CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(inputs2.get(time).getMem()));
         CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(input2.getMem()));
         
         CL.clEnqueueNDRangeKernel(queue, kernel, 1, null, input2.global_size, input2.local_size, 0, null, null);
      } else {
         
      }
   }
   
   protected void prepare() {
      input = Tensor.create(outputSize);
      input2 = Tensor.create(outputSize);
      output = Tensor.create(outputSize);
      
      if(OPEN_CL) {
         if(!ready) {
            kernel = getKernel("./kernel/pointmul.c", "pointmul");
            ready = true;
         }
      }
   }
   
}