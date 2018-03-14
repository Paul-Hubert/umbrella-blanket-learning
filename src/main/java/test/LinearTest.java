package test;

import learning.tensor.*;
import learning.node.*;
import learning.optimize.*;
import java.util.Arrays;

import learning.ComputeContext;
import static learning.ComputeContext.*;

public class LinearTest {
   
   public static void test(boolean CL) {
      if(ComputeContext.init()) {
         ComputeContext.OPEN_CL = CL;
      }
      
      int[] iSize = {128*10}, hiddenSize = {128*10}, oSize = {128*10}, lSize = {128};
      
      Input input = new Input(iSize), label = new Input(lSize);
      SoftmaxLoss softmax = (SoftmaxLoss)input.chain(new Linear(hiddenSize)).chain(new Sigmoid()).chain(new Linear(oSize)).chain(new SoftmaxLoss(label));
      
      Tensor inputs = Tensor.create(iSize), labels = Tensor.create(lSize);
      inputs.loadRandom(0f,1f);
      labels.loadRandom(0f,1f);
      
      input.load(inputs);
      label.load(labels);
      
      //0.001, 0.9, 0.999, 0.00000001
      
      AdamOptimizer gd = new AdamOptimizer(0.001f, 0.9f, 0.999f, 0.00001f);
      long time = System.currentTimeMillis();
      for(int i = 0; i<1000; i++) {
         //label.open();
         //input.open();
      }
      System.out.println("Completed 1000 iterations of linear-sigmoid-linear-softmax-loss of size " + iSize[0] + " in rows of " + (iSize[0]/lSize[0]) +
      " in " + (System.currentTimeMillis() - time) + " milliseconds");
      
      if(OPEN_CL) {
         ComputeContext.release();
      }
   }
   
}
