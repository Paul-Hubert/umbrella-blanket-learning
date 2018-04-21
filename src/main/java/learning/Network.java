package learning;

import java.util.ArrayList;

import learning.tensor.Tensor;
import learning.optimize.Optimizer;
import learning.operation.*;
import learning.node.*;

public class Network {
   
   private static Network current;
   
   private boolean RECURRENT = false;
   
   private ArrayList<Input> inputs = new ArrayList<>();
   private ArrayList<Output> outputs = new ArrayList<>();
   private ArrayList<RecurrentGate> recurrent = new ArrayList<>();
   
   private final ForwardOperation forward;
   private final BackwardOperation backward;
   private final OptimizeOperation optimize;
   
   private Operation op;
   
   private int passCount = 0;
   
   public Network() {
      forward = new ForwardOperation();
      backward = new BackwardOperation();
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
      if(n instanceof RecurrentGate) {
         recurrent.add((RecurrentGate) n);
         RECURRENT = true;
      } else if(n instanceof Input) {
         inputs.add((Input) n);
      } else if(n instanceof Output) {
         outputs.add((Output) n);
      }
   }
   
   public void operate(Node n) {
      op.operate(n);
   }
   
   public Node getLast() {
      return op.getLast();
   }
   
   public Tensor calculate(Node n) {
      op = forward;
      forward.prepare();
      return n.forwardOp();
   }
   
   public Tensor forwardPass(Node n, Operation operation) {
      op = operation;
      op.prepare();
      return n.forwardOp();
   }
   
   public void gradientsFrom(Node n) {
      op = backward;
      backward.prepare();
      Tensor t = Tensor.create(n.getOutputSize()).loadOnes();
      n.backwardOp(t);
   }
   
   public void optimize(Node n, Optimizer opt) {
      op = optimize;
      optimize.setOptimizer(opt);
      optimize.prepare();
      n.backwardOp(null);
   }
   
   public void backwardPass(Node n, Tensor t, Operation operation) {
      op = operation;
      op.prepare();
      n.backwardOp(t);
   }
   
}