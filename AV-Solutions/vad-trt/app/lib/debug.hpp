#pragma once
#include "vad_model.hpp"
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace autoware::tensorrt_vad {

class FileVadLogger : public VadLogger {
private:
    mutable std::ofstream ofs_;
    std::string log_path_;

    std::string get_timestamp() const {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm;
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y%m%d-%H:%M:%S");
        return oss.str();
    }

    void write(const std::string& level, const std::string& message) const {
        if (ofs_.is_open()) {
            ofs_ << "[" << level << "] " << message << std::endl;
        }
    }

public:
    FileVadLogger() {
        std::filesystem::create_directories("log");
        log_path_ = "log/log_" + get_timestamp() + ".txt";
        ofs_.open(log_path_, std::ios::out | std::ios::app);
    }
    ~FileVadLogger() override {
        if (ofs_.is_open()) ofs_.close();
    }
    void debug(const std::string& message) const override { write("DEBUG", message); }
    void info(const std::string& message) const override { write("INFO", message); }
    void warn(const std::string& message) const override { write("WARN", message); }
    void error(const std::string& message) const override { write("ERROR", message); }
    std::string log_path() const { return log_path_; }
};

} // namespace autoware::tensorrt_vad 