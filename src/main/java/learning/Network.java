package learning;

import java.util.ArrayList;

import learning.tensor.Tensor;
import learning.optimize.Optimizer;
import learning.operation.*;
import learning.node.*;

public class Network {
   
   private static Network current;
   
   private ArrayList<Input> inputs = new ArrayList<>();
   
   private final Operation forward, backward;
   private final OptimizeOperation optimize;
   
   public boolean RECURRENT = false;
   
   public Network() {
      forward = new Operation();
      backward = new Operation();
      optimize = new OptimizeOperation();
      current = this;
   }
   
   public void on() {
      current = this;
   }
   
   public static Network getCurrent() {
      return current;
   }
   
   public void add(Node n) {
      if(n instanceof Input && !inputs.contains((Input) n)) {
         inputs.add((Input) n);
      }
      if(n instanceof RecurrentGate) {
         RECURRENT = true;
      }
   }
   
   public Tensor calculate(Node n) {
      return n.forwardOp(forward);
   }
   
   public void gradientsFrom(Node n) {
      Tensor t = Tensor.create(n.getOutputSize()).loadOnes();
      n.backwardOp(t, backward);
   }
   
   public void optimize(Node n, Optimizer opt) {
      opt.prepare();
      optimize.setOptimizer(opt);
      n.backwardOp(null, optimize);
   }
   
}