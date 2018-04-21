package learning.node;

import learning.tensor.Tensor;
import learning.operation.Operation;

import org.jocl.*;
import static org.jocl.CL.*;
import static learning.ComputeContext.*;

public class Concatenate extends I2Node {
   
   private cl_kernel kernel, kernel2;
   
   private int[] offset;
   
   public void forwardProp() {
      
      if(OPEN_CL) {
         CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(in.getMem()));
         CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(in2.getMem()));
         CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(output.getMem()));
         CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(offset));
         
         CL.clEnqueueNDRangeKernel(queue, kernel, 1, null, output.global_size, output.local_size, 0, null, null);
         
      } else {
         for(int v = 0; v<output.vectorNum(); v++) {
            float[] vecO = output.getVector(v), vecI1 = in.getVector(v), vecI2 = in2.getVector(v);
            System.arraycopy(vecI1, 0, vecO, 0, vecI1.length);
            System.arraycopy(vecI2, 0, vecO, vecI1.length, vecI2.length);
         }
      }
   }
   
   public void backwardProp() {
      
      if(OPEN_CL) {
         
         CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(dO.getMem()));
         CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(input.getMem()));
         CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(input2.getMem()));
         CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(offset));
         
         CL.clEnqueueNDRangeKernel(queue, kernel2, 1, null, output.global_size, output.local_size, 0, null, null);
         
      } else {
         for(int v = 0; v<dO.vectorNum(); v++) {
            float[] vecdO = dO.getVector(v), vecdI1 = input.getVector(v), vecdI2 = input2.getVector(v);
            System.arraycopy(vecdO, 0, vecdI1, 0, vecdI1.length);
            System.arraycopy(vecdO, vecdI1.length, vecdI2, 0, vecdI2.length);
         }
      }
   }
   
   protected void setOutputSize() {
      outputSize = new int[before.getOutputSize().length];
      outputSize[0] = before.getOutputSize()[0] + before2.getOutputSize()[0];
      int bufferSize = before.getOutputSize()[0];
      for(int i = 1; i<outputSize.length; i++) {
         outputSize[i] = before.getOutputSize()[i];
         bufferSize *= before.getOutputSize()[i];
      }
      offset = new int[] {bufferSize};
   }
   
   protected void prepare() {
      input = Tensor.create(before.getOutputSize());
      input2 = Tensor.create(before2.getOutputSize());
      output = Tensor.create(outputSize);
      
      if(OPEN_CL) {
         if(!ready) {
            kernel = getKernel("./kernel/concatenate.c", "concatenate");
            kernel2 = getKernel("./kernel/dconcatenate.c", "dconcatenate");
            ready = true;
         }
      }
   }
}