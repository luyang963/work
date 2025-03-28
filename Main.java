public class Main {
    public static void main(String[] args) {
        Direction current = Direction.NORTH;
        System.out.println("Facing: " + current);
        System.out.println("Left: " + current.left());  // WEST
        System.out.println("Right: " + current.right()); // EAST
    }
}
