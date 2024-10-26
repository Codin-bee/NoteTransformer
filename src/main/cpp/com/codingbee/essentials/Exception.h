#include <string>
#ifndef EXCEPTION_H
#define EXCEPTION_H

enum ExceptionType{
    INVALID_ARGUMENT,
    INNER_ERROR,
    FILE_HANDLEING,
    UNKOWN
    };

class Exception{
    private:
    std::string message;
    ExceptionType type;
    public:
    Exception(std::string msg){ 
    message = msg;
    }
    Exception(ExceptionType excType){
        type = excType;
    }
    Exception(std::string msg, ExceptionType excType){
        message = msg;
        type = excType;
    }

    std::string getMessage(){
        return message;
    }

    ExceptionType getType(){
        return type;
    }
};
#endif