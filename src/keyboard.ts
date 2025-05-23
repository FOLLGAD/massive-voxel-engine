export class KeyboardState {
  downKeys: Set<string>;
  pressedKeys: Set<string>;
  mouseDown: boolean;
  mouseClicked: boolean;
  mouseRightClicked: boolean;

  constructor() {
    this.downKeys = new Set<string>();
    this.pressedKeys = new Set<string>();
    this.mouseDown = false;
    this.mouseClicked = false;
    this.mouseRightClicked = false;
  }
}
