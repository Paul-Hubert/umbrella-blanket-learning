package learning;

import java.util.ArrayList;

import learning.tensor.Tensor;
import learning.optimize.Optimizer;
import learning.operation.*;
import learning.node.*;

public class Network {
   
   private static Network current;
   
   private boolean RECURRENT = false;
   private int time = 0, opID = 0;
   
   private ArrayList<Input> inputs = new ArrayList<>();
   private ArrayList<Output> outputs = new ArrayList<>();
   private ArrayList<RecurrentGate> recurrent = new ArrayList<>();
   private ArrayList<Node> nodes = new ArrayList<>();
   
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
      nodes.add(n);
   }
   
   public void operate(Node n) {
      op.operate(n);
   }
   
   public Node getLast() {
      return op.getLast();
   }
   
   public int getTime() {
      return time - 1;
   }
   
   public int getOpID() {
      return opID - 1;
   }
   
   public Tensor calculate(Node n) {
      return forwardPass(n, forward);
   }
   
   public Tensor forwardPass(Node n, Operation operation) {
      op = operation;
      opID++;
      op.prepare();
      time++;
      return n.forwardOp();
   }
   
   public void gradientsFrom(Node n) {
      Tensor t = Tensor.create(n.getOutputSize()).loadOnes();
      backwardPass(n, t, backward);
      time--;
   }
   
   public void optimize(Node n, Optimizer opt) {
      optimize.setOptimizer(opt);
      backwardPass(n, null, optimize);
      time = 0;
   }
   
   public void backwardPass(Node n, Tensor t, Operation operation) {
      op = operation;
      opID++;
      op.prepare();
      if(!(n instanceof Output)) System.out.println("Node must be Output");
      n.backwardOp(t);
      for(Output o : outputs) {
         if(o != n) {
            o.backwardOp(Tensor.zero(o.getOutputSize()));
         }
      }
      
      for(RecurrentGate o : recurrent) {
         if(o != n) {
            o.backwardOp(Tensor.zero(o.getOutputSize()));
         }
      }
   }
   
}