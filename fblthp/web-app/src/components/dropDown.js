import React from "react"
import {Link} from "react-router-dom"
import './dropDown.css'

/*
    Following tutorial from Codemzy
    https://www.codemzy.com/blog/reactjs-dropdown-component
*/


const DropdownContext = React.createContext({
    open: false,
    setOpen: () =>{},
})

function Dropdown({children, ...props}) {
    const [open, setOpen] = React.useState(false)

    return(
        <DropdownContext.Provider value ={{open, setOpen}}>
            <div className="dropdown">{children}</div>
        </DropdownContext.Provider>
    )

}

function DropdownButton({children, ...props}){
    const {open, setOpen} = React.useContext(DropdownContext)

    function toggleOpen(){
        setOpen(!open)
    }

    return (
        <button onClick={toggleOpen} className="dropdownButton">
            { children }
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" width={15} height={15} strokeWidth={4} stroke="currentColor" className={`menuIcon ${open ? "rotate-180" : "rotate-0"}`}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
            </svg>
        </button>
    )
}

Dropdown.Button = DropdownButton

function DropdownContent({children}){
    const {open} = React.useContext(DropdownContext)

    return(
        <div className={`content ${ open ? "shadow-md" : "hidden"}`}>
            { children }
        </div>
    )
}

Dropdown.Content = DropdownContent

function DropdownList({children, ...props}){
    const {setOpen} = React.useContext(DropdownContext)

    return (
        <ul onClick={()=>setOpen(false)} className="list" {...props}>
            {children}
        </ul>
    )
}

Dropdown.List = DropdownList

function DropdownItem({children, ...props}){
    /*
    */
    return(
        <li>
            <Link className="item" {...props}>
                {children}
            </Link>
            
        </li>
    )
}

Dropdown.Item = DropdownItem

export default Dropdown