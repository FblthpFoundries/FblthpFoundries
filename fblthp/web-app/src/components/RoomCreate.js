import React from "react"
import constants from "../constants"
import './RoomCreate.css'

function RoomCreate({enter}){
    const [sets, setSets] = React.useState([])
    const [selection, select] = React.useState([null,null])
    React.useEffect(()=>{
        fetch(constants['cardServer'] + '/getSets',
            {mode: 'cors',
            method: 'get', 
            headers: { 'Access-Control-Allow-Origin': constants['cardServer'],}
            })
                .then(response => response.json())
                .then(data => setSets(data))
                .catch(e => console.log(e))}
    ,[])

    const createFunc = () => {
        const data = {setId: selection[1]}
        fetch(constants['cardServer'] + '/create',
            {mode:'cors',
            method:'post',
            headers: {
                'Access-Control-Allow-Origin': constants['cardServer'],
                'Content-Type': 'application/json'
            },
            body : JSON.stringify(data)}
        ).then(response => response.json())
        .then(data => enter(data, true))
        .catch(e => console.log(e))
    }

    return(
        <>
        <ul style={{maxHeight:200, overflow:'auto'}} className="setList">
           {sets.map( s => 
            <li className="listElm" key={s['code']} onClick={() => select([s['name'], s['code']])}>
                {s['name']}
            </li>)} 
        </ul> 
        <div className="selection">
            {selection[0]? <p >Set: {selection[0]}</p>: <p >Please make a selection</p>}
        </div>
        {selection[0]?<button onClick={createFunc}>Create</button>:<></>}
        </>
    )
}

export default RoomCreate