package src.main.java.com.codingbee;
public class MidiNote {
    private int key;
    private int velocity;
    private long position;
    private long ticksToPrev;
    private long ticksToNext;

    public MidiNote(int key, int velocity, long position, long ticksToPrev, long ticksToNext) {
        this.key = key;
        this.velocity = velocity;
        this.position = position;
        this.ticksToPrev = ticksToPrev;
        this.ticksToNext = ticksToNext;
    }

    public int getKey() {
        return key;
    }

    public void setKey(int key) {
        this.key = key;
    }

    public int getVelocity() {
        return velocity;
    }

    public void setVelocity(int velocity) {
        this.velocity = velocity;
    }

    public long getPosition() {
        return position;
    }

    public void setPosition(long position) {
        this.position = position;
    }

    public long getTicksToPrev() {
        return ticksToPrev;
    }

    public void setTicksToPrev(long ticksToPrev) {
        this.ticksToPrev = ticksToPrev;
    }

    public long getTicksToNext() {
        return ticksToNext;
    }

    public void setTicksToNext(long ticksToNext) {
        this.ticksToNext = ticksToNext;
    }

    @Override
    public String toString() {
        return "MidiNote{" +
                "key=" + key +
                ", velocity=" + velocity +
                ", absolutePosition=" + position +
                ", ticksToPreviousNote=" + ticksToPrev +
                ", ticksToNextNote=" + ticksToNext +
                '}';
    }

    public String toInputSaveFormat() {
        return key + " " + velocity + " " + ticksToPrev + " " + ticksToNext + " " + position;
    }

    public String toOutputSaveFormat() {
        String outputFormat = "";
        for (int i = 0; i < 128; i++){
            if (i == key){
                outputFormat += "1 ";
            }else{
                outputFormat += "0 ";
            }
        }
        for (int i = 0; i < 128; i++){
            if (i == velocity){
                outputFormat += "1 ";
            }else{
                outputFormat += "0 ";
            }
        }
        outputFormat += ticksToPrev + " " + ticksToNext + " " + position;
        return outputFormat;
    }
}
