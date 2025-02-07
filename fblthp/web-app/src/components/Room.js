import React from "react"
import {io} from "socket.io-client"
import constants from "../constants"
import './Room.css'

/*
* 1) connect to server with socket, at this point client is in room but room may not be started
* 2) Once room is started, client will have two states: waiting for a pack or choosing from a pack
*/

const socket = io(constants['cardServer'], {autoConnect: false})
var connected = false
var roomStarted = false

function PackGrid({ pack, onPick }) {
    function pick(id) {
        socket.emit('pick', id)
        onPick()  
    }

    return (
        <div className="cardGrid">
            {pack.map((card) => {
                return (
                    <div className="card" key={card['id']}>
                        <button className="button" onClick={() => pick(card['id'])}>{card['name']}</button>
                    </div>
                )
            })}
        </div>
    )
}

function DraftRoom() {
    const [roomState, setRoomState] = React.useState(roomStarted)
    const [hasPack, setHasPack] = React.useState(false)
    const [pack, setPack] = React.useState([])

    function pickedCard() {
        setHasPack(false)
        setPack([])
    }

    function updateRoom(state) {
        roomStarted = state
        setRoomState(state)
    }

    React.useEffect(() => {
        socket.on('roomStarted', () => {
            updateRoom(true)
        })

        socket.on('pack', (data) => {
            setPack(data['pack'])
            setHasPack(true)
        })
    }, [])

    const displayPack = hasPack ? <PackGrid pack={pack} onPick={pickedCard} /> : <p>waiting for pack</p>

    return (<>
        {roomState ? displayPack : <p>Waiting for room to start</p>}
    </>)
}

function Room() {
    const [connectHook, setConnectHook] = React.useState(connected)

    function updateConnect(state) {
        connected = state
        setConnectHook(state)
    }

    React.useEffect(() => {
        if (!connected) {
            socket.connect()
            updateConnect(true)
        }
    }, [])

    return (
        <>
            Testing Room
            {connectHook ? <DraftRoom /> : <p>Not Connected</p>}
        </>
    )
}

export default Room