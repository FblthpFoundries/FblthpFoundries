import React from "react"
import Room from './Room.js'
import './Draft.css'
import RoomModal from './Modal.js'
import RoomCreate from "./RoomCreate"
import JoinRoom from "./JoinRoom.js"


function Welcome({enter}){
    return(
        <div className="welcome">
            <RoomModal
                buttonText = 'Create Room'
                label = 'Create Room'
                style={{display:'flex'}}
            >
                <RoomCreate enter = {enter}/>
            </RoomModal>
            <RoomModal
                buttonText = 'Join Room'
                label = 'Join Room'
                Style = {{display:'flex'}}
            >
                <JoinRoom enter = {enter}/>

            </RoomModal>
        </div>
    )
}

function Draft(){
    const [roomId, setRoomId] = React.useState(null)
    const [inRoom, setInRoom] = React.useState(false)
    const [isHost, setIsHost] = React.useState(false)
    React.useEffect(()=>{setInRoom(false)}, [])
    return(
        <>
          {inRoom ?<Room roomId = {roomId} isHost = {isHost}/>: <Welcome enter={(id, host)=>{setRoomId(id);setInRoom(true);setIsHost(host)}}/> }  
        </>
    )
}

export default Draft