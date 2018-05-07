package learning;

import learning.tensor.*;
import learning.node.*;
import learning.optimize.*;

import test.*;

import static learning.ComputeContext.*;

public class Main {
   
   public static void main(String[] args) {
      
      for (String arg : args) {
         if (arg.equals("-noCL")) {
            OPEN_CL = false;
         } else if (arg.equals("LinearTest")) {
            LinearTest.test();
            return;
         } else if (arg.equals("MNIST")) {
            MNIST.test();
            return;
         } else if (arg.equals("ShakeLSTM")) {
            ShakeLSTM.test();
            return;
         } else if (arg.equals("devices")) {
            JOCLDeviceQuery.test();
            return;
         }
      }
      
      if(OPEN_CL) {
         ComputeContext.init();
      }
      
      Mastermind game = new Mastermind();
      Network net = new Network();
      
      int[] iSize = {12}, hSize = {400}, oSize = {40}, lSize = {5}, fSize = {8,5};
      
      Input input = new Input(iSize), label = new Input(lSize);
      RecurrentGate h = new RecurrentGate(hSize), C = new RecurrentGate(hSize);
      
      Copy copy = (Copy) input.chain(h.chain(new Concatenate())).chain(new Copy());
      
      PointMul forgetGate = (PointMul) copy.chain(new Linear(hSize).name("f")).chain(new Sigmoid()).chain(new PointMul());
      copy = (Copy) copy.chain(new Copy());
      PointMul addingWeight = (PointMul) copy.chain(new Linear(hSize).name("w")).chain(new Sigmoid()).chain(new PointMul());
      copy = (Copy) copy.chain(new Copy());
      PointAdd addingGate = (PointAdd) copy.chain(new Linear(hSize).name("a")).chain(new Tanh()).chain(addingWeight).chain(new PointAdd());
      PointMul outputGate = (PointMul) copy.chain(new Linear(hSize).name("o")).chain(new Sigmoid()).chain(new PointMul());
      
      copy = (Copy) C.chain(forgetGate).chain(addingGate).chain(new Copy());
      copy.chain(C);
      copy = (Copy) copy.chain(new Tanh()).chain(outputGate).chain(new Linear(oSize).name("out")).chain(new Copy());
      copy.chain(h);
      SoftmaxLoss ce = (SoftmaxLoss) copy.chain(new Shape(fSize)).chain(new SoftmaxLoss(label));
      
      Output cross = (Output) ce.chain(new Output());
      
      
      Tensor inputs = Tensor.create(iSize), labels = Tensor.create(lSize);
      
      input.load(inputs);
      label.load(labels);
      
      //0.001, 0.9, 0.999, 0.00000001
      
      AdamOptimizer gd = new AdamOptimizer(0.001f, 0.9f, 0.999f, 0.00001f);
      long time = 0;
      for(int i = 0; i<1000000; i++) {
         game.init();
         
         float[] inputsB = new float[iSize[0]];
         
         inputs.loadRandom(0f,1f);
         
         net.calculate(cross);
         
         net.gradientsFrom(ce);
         
         net.optimize(ce, gd);
         
         if(i%1000 == 0) {
            System.out.println(System.currentTimeMillis() - time);
            time = System.currentTimeMillis();
         }
         
      }
      
      if(OPEN_CL) {
         ComputeContext.release();
      }
   }
   
}
