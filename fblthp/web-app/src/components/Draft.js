import React from "react"
import Room from './Room.js'
import './Draft.css'
import RoomModal from './Modal.js'
import RoomCreate from "./RoomCreate"


function Welcome({enter}){
    return(
        <div className="welcome">
            <RoomModal
                buttonText = 'Create Room'
                label = 'Create Room'
                style={{display:'flex'}}
            >
                <RoomCreate/>
                <button onClick={enter}>Create </button>
            </RoomModal>
            <button onClick={enter}>Join Room</button>
        </div>
    )
}

function Draft(){
    const [inRoom, setInRoom] = React.useState(false)
    React.useEffect(()=>{setInRoom(false)}, [])
    return(
        <>
          {inRoom ?<Room/>: <Welcome enter={()=>setInRoom(true)}/> }  
        </>
    )
}

export default Draft