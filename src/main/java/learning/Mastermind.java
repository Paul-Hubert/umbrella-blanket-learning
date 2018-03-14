package learning;

public class Mastermind {
   
   private int[] solution, solCount;
   
   public void init() {
      solution = new int[5];
      for(int i=0; i<5; i++) {
         solution[i] = (int) Math.floor(Math.random()*8);
      }
      solCount = count(solution);
   }
   
   public int[] guess(int[] guess) {
      int[] guessCount = count(guess);
      int wp = 0, sum = 0;
      for(int i = 0; i<8; i++) {
         if(i<5 && guess[i] == solution[i]) wp++;
         sum += Math.min(guessCount[i], solCount[i]);
      }
      return new int[] {wp,sum-wp};
   }
   
   private int[] count(int[] t) {
      int[] count = {0,0,0,0,0,0,0,0};
      for(int i = 0; i<5; i++) {
         count[t[i]]++;
      }
      return count;
   }
   
   public int[] getSolution() {
      return solution;
   }
   
}