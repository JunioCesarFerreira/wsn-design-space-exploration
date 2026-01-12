import logging
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

logger = logging.getLogger(__name__)

def update_simulation_xml(
    fixed_positions: list[tuple[float, float]],
    mobile_positions: list[tuple[float, float]],
    root_motes: list[int],
    simulation_time: float,
    tx_range: float,
    interference_range: float,
    input_file: str,
    output_file: str
) -> None:
    """Atualiza arquivo XML de simulação com novos parâmetros.
    
    Args:
        fixed_positions: Lista de tuplas (x, y) com posições fixas
        mobile_positions: Lista de tuplas (x, y) com posições iniciais dos móveis
        root_motes: Lista de IDs dos motes servidores
        simulation_time: Tempo de simulação em minutos
        tx_range: Alcance de transmissão
        interference_range: Alcance de interferência
        inputFile: Caminho do arquivo XML de entrada (template)
        outputFile: Caminho do arquivo XML de saída
    """
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Updates radio parameters
    radiomedium = root.find(".//radiomedium")
    if radiomedium is not None:
        transmitting_range = radiomedium.find("transmitting_range")
        if transmitting_range is not None:
            transmitting_range.text = str(tx_range)
        interference_range_elem = radiomedium.find("interference_range")
        if interference_range_elem is not None:
            interference_range_elem.text = str(interference_range)
    
    # Update simulation time in JS script keeping CDATA
    script_element = root.find(".//script")
    if script_element is not None and script_element.text is not None:
        script_text = script_element.text
        new_timeout = simulation_time * 60000  # Convertendo minutos para milissegundos
        script_text = script_text.replace("const timeOut = X * 1000;", f"const timeOut = {new_timeout} * 1000;")
        script_text = script_text.replace("TIMEOUT(X);", f"TIMEOUT({new_timeout + 11000});") # 11 segundos de tolerância para fechar, pois o tempo de ping é 10 segundos.
        script_element.text = f"<![CDATA[\n{script_text}\n]]>"
    
    # update motes
    motetype_root = root.find(".//motetype[description='server']")
    motetype_client = root.find(".//motetype[description='client']")
    
    if motetype_root is not None:
        for mote in motetype_root.findall("mote"):
            motetype_root.remove(mote)
    if motetype_client is not None:
        for mote in motetype_client.findall("mote"):
            motetype_client.remove(mote)
    
    for i, (x, y) in enumerate(fixed_positions + mobile_positions):
        mote_type = motetype_root if i + 1 in root_motes else motetype_client
        if mote_type is not None:
            mote = ET.SubElement(mote_type, "mote")
            
            interface_config = ET.SubElement(mote, "interface_config")
            interface_config.text = "org.contikios.cooja.interfaces.Position"
            ET.SubElement(interface_config, "pos", x=str(x), y=str(y))
            
            id_config = ET.SubElement(mote, "interface_config")
            id_config.text = "org.contikios.cooja.contikimote.interfaces.ContikiMoteID"
            ET.SubElement(id_config, "id").text = str(i + 1)
            
    if mobile_positions is None or len(mobile_positions) == 0:
        # Remove o plugin de mobilidade, se existir
        for plugin in root.findall(".//plugin"):
            if plugin.text and "org.contikios.cooja.plugins.Mobility" in plugin.text:
                root.remove(plugin)
    
    xml_str = ET.tostring(root, encoding='utf-8')
    parsed_xml = minidom.parseString(xml_str)
    with open(output_file, "w", encoding="utf-8") as f:
        output = parsed_xml.toprettyxml(indent="  ")
        output = output.replace("?>", "encoding=\"UTF-8\"?>")
        output = output.replace("&gt;", ">")
        output = output.replace("&lt;", "<")
        output = output.replace("&quot;", "\"")
        output = output.replace("<![CDATA[\n", "<![CDATA[")
        output = output.replace("\n]]>", "]]>")
        
        # Remove blank lines, exceto dentro de CDATA
        inside_cdata = False
        lines_without_blanks = []
        for line in output.splitlines():
            if "<![CDATA[" in line:
                inside_cdata = True
            if inside_cdata or line.strip():
                lines_without_blanks.append(line)
            if "]]>" in line:
                inside_cdata = False
                
        final_content = "\n".join(lines_without_blanks)
        f.write(final_content)
    
    logger.info(f"File {output_file} generated successfully!")