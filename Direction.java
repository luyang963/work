public enum Direction {
    NORTH("North"),
    SOUTH("South"),
    EAST("East"),
    WEST("West");

    private String name;

    Direction(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public Direction left() {
        switch (this) {
            case NORTH:
                return WEST;
            case WEST:
                return SOUTH;
            case SOUTH:
                return EAST;
            case EAST:
                return NORTH;
            default:
                return this; // This should never be reached
        }
    }

    public Direction right() {
        switch (this) {
            case NORTH:
                return EAST;
            case EAST:
                return SOUTH;
            case SOUTH:
                return WEST;
            case WEST:
                return NORTH;
            default:
                return this; // This should never be reached
        }
    }
}
