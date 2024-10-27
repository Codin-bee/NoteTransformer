package src.main.java.com.codingbee;

import javax.sound.midi.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

public class MidiGenerator {
    public static void main(String[] args) {
        int[][] matrix = loadMatrix("C:\\Users\\theco\\NoteTransformers\\Experiment\\outTrack.txt");
        createMidi(matrix, "C:\\Users\\theco\\NoteTransformers\\Experiment\\firstSong.midi");
    }

    private static int[][] loadMatrix(String path){
        int[][] matrix = null;
        try(BufferedReader reader = new BufferedReader(new FileReader(path))){
            String[] header = reader.readLine().split(" ");
            int rows = Integer.parseInt(header[0]);
            int columns = Integer.parseInt(header[1]);
            matrix = new int[rows][columns];
            for (int i = 0; i < rows; i++){
                String[] lineValues = reader.readLine().split(" ");
                for (int j = 0; j < columns; j++){
                    matrix[i][j] = Integer.parseInt(lineValues[j]);
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return matrix;
    }

    public static void createMidi(int[][] noteValues, String fileName) {
        try {
            Sequence sequence = new Sequence(Sequence.PPQ, 480);
            Track track = sequence.createTrack();

            int currentTick = 0;

            for (int[] note : noteValues) {
                int key = note[0];
                int velocity = note[1];
                int tickDistance = note[2];
                currentTick += tickDistance;
                track.add(createNoteEvent(ShortMessage.NOTE_ON, key, velocity, currentTick));
                track.add(createNoteEvent(ShortMessage.NOTE_OFF, key, 0, currentTick + 100));
            }

            File midiFile = new File(fileName);
            MidiSystem.write(sequence, 1, midiFile);

            System.out.println("MIDI file created successfully at path: " + fileName);

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static MidiEvent createNoteEvent(int command, int key, int velocity, int tick)
            throws InvalidMidiDataException {
        ShortMessage message = new ShortMessage();
        message.setMessage(command, 0, key, velocity);
        return new MidiEvent(message, tick);
    }
}
