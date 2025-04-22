SELECT 
    txm.message_id,
    txm.tag_id,
    t.name,
    t.visible,
    m.text,
    m.longitude,
    m.latitude,
    m.area_id,
    m.author_user_id,
    m.device_string,
    m.address,
    m.tags_fulltext
FROM 
    tag_x_message txm
JOIN 
    tag t ON txm.tag_id = t.id
JOIN 
    message m ON txm.message_id = m.id
WHERE 
	m.visible=1
ORDER BY 
    txm.message_id, txm.tag_id