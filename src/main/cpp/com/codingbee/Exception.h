#include <string>
#ifndef EXCEPTION_H
#define EXCEPTION_H

class Exception{
    private:
    std::string message;
    public:
    Exception(std::string msg){ 
    message = msg;
    }

    std::string getMessage(){
        return message;
    }
};
#endif