#include <iostream>
#include "NoteTransformer.h"
#include "FileUtils.h"
#include "Exception.h"

using namespace std;

int main(){
    cout << "NOTE TRANSFORMER TRAINING" << "\n" << 0.021;
    cout << "Randomly initialize the transformer?" << "\n";
    system("pause");
    NoteTransformer transformer(128, 4, 2, 16, 16, 2, 2, 2, 64, 32, 64);
    transformer.randomInit();
    //cout << "Done \n" << "Test saving and initialization from files? \n";
    //system("pause");
    try{
    //transformer.save("C:/Users/theco/NoteTransformers/No1");
    //transformer.~NoteTransformer();
    //transformer.init("C:/Users/theco/NoteTransformers/No1");
    cout << "The model has been initialized with " << transformer.getNumberOfParameters() << " parameters \n";
    cout << "Start the training? \n";
    system ("pause");
    TrainingSettings settings = TrainingSettings();
    settings.setDataPath("C:\\Users\\theco\\NoteTransformers\\Datasets\\haydn1\\");
    settings.setBatchSize(1);
    settings.setEpochs(1);
    settings.setLearningRate(0.002);
    settings.setSoftmaxTemperature(1.1);
    settings.setBeta_1(0.9);
    settings.setBeta_2(0.98);
    settings.setEpsilon(0.00000001);

    int** processedValues = FileUtils::readIntMatrixFromFile("C:\\Users\\theco\\NoteTransformers\\Datasets\\haydn1\\input0.txt");
    cout << "Loading fine \n";
    float** result =  transformer.process(processedValues);
    for (int i = 0; i < 128; i++){
        for (int j = 0; j < 5; j++){
            cerr << result[i][j] << " ";
        }
        cerr << "\n";
    }
    //transformer.train(settings);
    cerr << "\n \n SO";
    for (int i = 0; i < 128; i++){
        for (int j = 0; j < 5; j++){
            cerr << processedValues[i][j] << " ";
        }
        cerr << "\n";
    }
    }catch (Exception e){
        cerr << e.getMessage();
    }catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    } catch (const std::system_error& e) {
        cout << "Caught exception: " << e.what() << "\n";
    }
    
    cout << "Training process has been finished." << "\n";
    cout << "Want to close this dialog and dealocate memory?" << "\n";
    system("pause");
    transformer.~NoteTransformer();
    return 0;
}