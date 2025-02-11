import React from "react"
import {io} from "socket.io-client"
import constants from "../constants"
import './Room.css'

/*
* 1) connect to server with socket, at this point client is in room but room may not be started
* 2) Once room is started,[] client will have two states: waiting for a pack or choosing from a pack
*/

const socket = io(constants['cardServer'], {autoConnect: false})
var connected = false
var roomStarted = false

function CardImage({card, pick}){
    const [cImage, setImage] = React.useState(card['img'])

    if( card['doubleFaced'])
        return(
            <img src={cImage} alt={card['name']}
            onClick={() => {pick(card['id'])}}
            onMouseEnter={()=>setImage(card['back'])}
            onMouseLeave={()=>setImage(card['img'])}
            key={card['id']} className='card'/>)



    return(
            <img src={card['img']} alt={card['name']}
            onClick={() => {pick(card['id'])}}
            key={card['id']} className='card'/>
        
   )
}

function PackGrid({ pack, onPick }) {
    function pick(id) {
        onPick()  
        socket.emit('pick', id)
    }

    return (
        <div className="cardGrid">
            {pack.map((card) => {
                return (
                    <CardImage card={card} pick={pick}/>
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