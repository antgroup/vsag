

#include "receiver.h"

void
Receiver::Init() {
    this->receive_first_message();
    read_stream_buf_ = new ReadStreamBuf(cv_struct_, total_size_, send_queue_, recv_queue_);
    read_stream_ = new std::istream(read_stream_buf_);
}

void
Receiver::receive_first_message() {
    while (true) {
        if (this->cv_struct_.pause_signal_.load()) {
            std::unique_lock<std::mutex> lk(this->cv_struct_.mu_);
            this->cv_struct_.cv_.wait(lk,
                                      [this] { return not this->cv_struct_.pause_signal_.load(); });
        }
        Message* message = this->recv_queue_->ReceiveMessage();
        if (message == nullptr) {
            continue;
        }
        this->total_size_ = message->size;
        break;
    }
}

void
Receiver::send_last_message() {
    Message* message = new Message();
    message->type = MessageType::kEnd;
    this->send_queue_->SendMessage(message);
}

void
Receiver::Run() {
    if (work_thread_ == nullptr) {
        work_thread_ = new std::thread([this]() { this->work(); });
    }
    {
        std::unique_lock<std::mutex> lk(cv_struct_.mu_);
        cv_struct_.pause_signal_.store(false);
    }
    cv_struct_.cv_.notify_one();
}

void
Receiver::Pause() {
    {
        std::unique_lock<std::mutex> lk(cv_struct_.mu_);
        cv_struct_.pause_signal_.store(true);
    }
    cv_struct_.cv_.notify_one();
}

void
Receiver::work() {
    if (status_ == ReceiverStatus::kInit) {
        this->Init();
        this->status_ = ReceiverStatus::kDataReceive;
    }
    if (status_ == ReceiverStatus::kDataReceive) {
        this->index_->Deserialize(*this->read_stream_);
        this->send_last_message();
        this->status_ = ReceiverStatus::kFinished;
    }
}
