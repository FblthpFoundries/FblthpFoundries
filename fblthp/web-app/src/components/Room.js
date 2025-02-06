import React from "react"
import {io} from "socket.io-client"
import constants from "../constants"
import { data } from "react-router-dom"


/*
* 1) connect to server with socket, at this point client is in room but room may not be started
* 2) Once room is started, client will have two states: waiting for a pack or choosing from a pack
*/

const socket = io(constants['cardServer'], {autoConnect: false})
var connected = false
var roomStarted = false


function PackGrid(pack){

    return(<p>pack</p>)
}

function DraftRoom(){
    const [roomState, setRoomState]= React.useState(roomStarted)
    const [hasPack, setHasPack] = React.useState(false)
    const pack = React.useRef(null)

    function updateRoom(state){
        roomStarted = state
        setRoomState(state)
    }

    React.useEffect(() => {
        socket.on('roomStarted', () => {
            updateRoom(true)
        })

        socket.on('pack', (data) => {
            pack.current = data
            setHasPack(true)    
        })
    }, [])

    const displayPack = hasPack ? <PackGrid pack/>: <p>waiting for pack</p>

    return(<>
        {roomState ? displayPack : <p>Waiting for room to start</p>}
    </>)
}

function Room(){

    const [connectHook, setConnectHook] = React.useState(connected)


    function updateConnect(state){
        connected = state
        setConnectHook(state)
    }

    React.useEffect(() => {
        if(!connected){
            socket.connect()
            updateConnect(true)
        }
    }, [])


    return(
        <>
            Testing Room
            {connectHook ? <DraftRoom/> : <p>Not Connected</p>}
        </>
    )
}

export default Room