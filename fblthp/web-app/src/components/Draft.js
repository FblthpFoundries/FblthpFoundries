import React from "react"
import Room from './Room.js'
import './Draft.css'


function Welcome({enter}){
    return(
        <div className="welcome">
            <button onClick={enter}>Create Room</button>
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