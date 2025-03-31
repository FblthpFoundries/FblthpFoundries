import React from "react"
import constants from "../constants"

function JoinRoom({enter}){
    const [roomId, setRoomId] = React.useState(-1)

    function checkRoom(isValid){
        if (isValid)
            enter(roomId, false)
        else
            alert('BAD ROOM ID MY GUY')
    }


    function onSubmit(event){
        event.preventDefault()
        fetch(constants['cardServer'] + '/checkRoom',
            {   mode: 'cors',
                method: 'post',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({'room': roomId})
            }
        ).then(response => response.json())
        .then(data => checkRoom(data['isValid']))
        .catch(e => console.log(e))
    }

    return(<>
    <form onSubmit={onSubmit}>
        <label>
            Enter Room ID:
            <input type='number' value={roomId} onChange={(e) => setRoomId(e.target.value)}/>
        </label>
        <input type='submit'/>
    </form>
    </>)
}

export default JoinRoom