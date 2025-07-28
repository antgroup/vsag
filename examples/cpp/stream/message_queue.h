

#pragma once

#include <cstdint>
#include <mutex>
#include <queue>

enum class MessageType {
    kStart,
    kData,
    kEnd,
};

struct Message {
    char data[4096];
    uint64_t size;
    MessageType type;
};

class MessageQueue {
public:
    MessageQueue() = default;
    ~MessageQueue() = default;

    void
    SendMessage(Message* message) {
        std::lock_guard<std::mutex> lock(mu_);
        message_queue_.push(message);
    }

    Message*
    ReceiveMessage() {
        std::lock_guard<std::mutex> lock(mu_);
        while (message_queue_.empty()) {
            return nullptr;
        }
        Message* message = message_queue_.front();
        message_queue_.pop();
        return message;
    }

private:
    std::queue<Message*> message_queue_;
    std::mutex mu_;
};