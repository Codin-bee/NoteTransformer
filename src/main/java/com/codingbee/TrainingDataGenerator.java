package src.main.java.com.codingbee;

import javax.sound.midi.*;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

public class TrainingDataGenerator {
    private static final int contextSize = 128;

    public static List<MidiNote> parseMidiFile(String filepath) {
        List<MidiNote> notes = new ArrayList<>();
        try {
            File midiFile = new File(filepath);
            Sequence sequence = MidiSystem.getSequence(midiFile);
            for (Track track : sequence.getTracks()) {
                long lastTick = 0;
                long position = 0;
                for (int i = 0; i < track.size(); i++) {
                    MidiEvent event = track.get(i);
                    MidiMessage message = event.getMessage();
                    if (message instanceof ShortMessage sm) {
                        if (sm.getCommand() == ShortMessage.NOTE_ON && sm.getData2() > 0) {
                            int key = sm.getData1();
                            int velocity = sm.getData2();
                            long currentTick = event.getTick();
                            long ticksToPrev = position == 0 ? 0 : (currentTick - lastTick);
                            MidiNote note = new MidiNote(key, velocity, position, ticksToPrev, 0);
                            notes.add(note);
                            lastTick = currentTick;
                            position++;
                        }
                    }
                }
            }
            for (int i = 0; i < notes.size() - 1; i++) {
                long ticksToNext = notes.get(i + 1).getTicksToPrev();
                notes.get(i).setTicksToNext(ticksToNext);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return notes;
    }

    public static void saveNoteList(String filePath, List<MidiNote> notes, boolean input) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))){
            writer.write(contextSize + " 5");
            if (input) {
                for (int i = 0; i < notes.size(); i++){
                    writer.newLine();
                    writer.write(notes.get(i).toInputSaveFormat());
                }
            }else{
                for (int i = 0; i < notes.size(); i++){
                    writer.newLine();
                    writer.write(notes.get(i).toOutputSaveFormat());
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) {
        String generatedDataFolder = "src/main/resources/data_preprocessed";
        List<MidiNote> notes;
        int exampleNo = -1;
        File[] midis = new File("src/main/resources/data_midis").listFiles();
        if (midis != null) {
            for (File midi : midis) {
                notes = parseMidiFile(midi.getPath());
                for (int i = 0; i + contextSize < notes.size(); i++) {
                    exampleNo++;
                    saveNoteList( generatedDataFolder + "/input" + exampleNo + ".txt", notes.subList(i, i + contextSize - 1), true);
                    saveNoteList( generatedDataFolder + "/output" + exampleNo + ".txt", notes.subList(i + 1, i + contextSize), false);
                }
            }
        }
    }
}