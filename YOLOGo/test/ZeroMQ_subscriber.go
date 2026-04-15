package main

import (
	"fmt"

	zmq "github.com/pebbe/zmq4"
)

func main() {
	subscriber, _ := zmq.NewSocket(zmq.SUB)
	defer subscriber.Close()
	
	subscriber.Connect("tcp://localhost:5555")
	subscriber.SetSubscribe("") // 모든 메시지 수신

	fmt.Println("Waiting for data from Python...")
	for {
		msg, _ := subscriber.RecvMessage(0) // 슬라이스로 반환됨 [topic, payload]
		topic := msg[0]
		payload := msg[1]
		fmt.Printf("[%s] %s\n", topic, payload)
	}
}