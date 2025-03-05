import React  from "react"
import Modal from "react-modal"
import PropTypes from 'prop-types'


function RoomModal({children, ...props}){
    console.log(props.buttonText)
    const [isOpen, setIsOpen] = React.useState(false)

    function openModal(){
        setIsOpen(true)
    }

    function closeModal(){
        setIsOpen(false)
    }


    return(
        <>
        <button onClick={openModal}>{props.buttonText}</button> 
        <Modal
            isOpen = {isOpen}
            onAfterOpen = {props.onOpen}
            onRequestClose = {closeModal}
            shouldCloseOnOverlayClick = {true}
        >
            {children}
        </Modal>
        </>
    )
}

RoomModal.propTypes = {
    buttonText : PropTypes.string.isRequired,
    onOpen : PropTypes.func.isRequired,
}

RoomModal.defaultProps = {
    buttonText: 'Open Modal',
    onOpen : () => {console.log("no on open")},
}

export default RoomModal