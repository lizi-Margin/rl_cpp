#pragma once
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <iostream>

namespace draw {

typedef std::unordered_map<std::string, float>  statistic;

class csv_drawer {
  public:
    inline csv_drawer(std::string filename) : filename(filename), mem(nullptr) {};

    inline void draw(statistic stat) {
      if (mem == nullptr) {
        init_mem(stat); 
        write_to_file();
      } else {
        append_mem(stat);
        append_to_file();
      }
    }
  private:
    inline void init_mem(statistic stat) {
      mem = std::make_unique<std::unordered_map<std::string, std::vector<float>>>();
      for (auto& [key, value] : stat) {
        std::vector<float> init_vec;
        init_vec.push_back(value);
        mem->emplace(key, std::move(init_vec));
      }
    }

    inline void append_mem(statistic stat) {
      //check if mem is initialized
      if (mem == nullptr) {
        throw std::runtime_error("mem is not initialized");
      }

      //check if stat has the same number of keys as mem
      if (stat.size() != mem->size()) {
        throw std::runtime_error("stat has a different number of keys than mem");
      }
      //check if stat has the same keys as mem
      for (auto& [key, value] : stat) {
        if (mem->find(key) == mem->end()) {
          throw std::runtime_error("stat has a key that is not in mem");
        }
      }
      
      for (auto& [key, value] : stat) {
        mem->at(key).push_back(value);
      }
    }

    inline void write_to_file() {
      if (file_exists()) {
        //红色
        std::cout << "\033[31m" << "file exists, remove old file" << "\033[0m" << std::endl;
        //mv old.csv to old.csv.bac and create new file
        std::string command = "mv " + filename + " " + filename + ".bac";
        int ret = system(command.c_str());
        if (ret != 0) {
          std::cout << "\033[31m" << "mv failed" << "\033[0m" << std::endl;
        }
        command = "touch " + filename;
        ret = system(command.c_str());
        if (ret!= 0) {
          std::cout << "\033[31m" << "touch failed" << "\033[0m" << std::endl;
        }
      }

      //check if mem is initialized
      if (mem == nullptr) {
        throw std::runtime_error("mem is not initialized");
      }
      //check if mem is empty
      if (mem->empty()) {
        throw std::runtime_error("mem is empty");
      }
      //check if mem has the same number of values for each key
      auto it = mem->begin();
      auto size = it->second.size();
      for (auto& [key, value] : *mem) {
        if (value.size()!= size) {
          throw std::runtime_error("mem has a different number of values for each key");
        }
      }
      //write to file
      std::ofstream file(filename);
      if (!file.is_open()) {
        throw std::runtime_error("could not open file");
      }
      //write header
      for (auto& [key, value] : *mem) {
        file << key << ",";
      }
      file << "\n";
      //write values
      for (size_t i = 0; i < size; i++) {
        for (auto& [key, value] : *mem) {
          file << value[i] << ",";
        }
        file << "\n";
      }
      file.close();
      clear_mem();
    }
    
    inline void append_to_file() {
      //check the file exists
      if (!file_exists()) {
        throw std::runtime_error("file not exists");
      }
      //check if mem is initialized
      if (mem == nullptr) {
        throw std::runtime_error("mem is not initialized");
      }
      //check if mem is empty
      if (mem->empty()) {
        throw std::runtime_error("mem is empty");
      }
      //check if mem has the same number of values for each key
      auto it = mem->begin();
      auto size = it->second.size();
      for (auto& [key, value] : *mem) {
        if (value.size()!= size) {
          throw std::runtime_error("mem has a different number of values for each key");
        }
      }
      //write to file
      std::ofstream file(filename, std::ios::app);
      if (!file.is_open()) {
        throw std::runtime_error("could not open file");
      }
      //write values
      for (size_t i = 0; i < size; i++) {
        for (auto& [key, value] : *mem) {
          file << value[i] << ",";
        }
        file << "\n";
      }
      file.close();
      clear_mem();
    }

    inline bool file_exists() {
      std::ifstream file(filename);
      // return file.good();
      return file.is_open();
    }
    
    inline void clear_mem() {
      for (auto& [key, value] : *mem) {
        value.clear();
      }
    }
  private:
    std::string filename;
    std::unique_ptr<std::unordered_map<std::string, std::vector<float>>> mem;
};

}